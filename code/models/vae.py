# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

from layers import GaussianSample, GaussianMerge, GumbelSoftmax
from inference import log_gaussian, log_standard_gaussian


class Perceptron(nn.Module):
    def __init__(self, dims, activation_fn=F.relu, output_activation=None):
        super(Perceptron, self).__init__()
        self.dims = dims
        self.activation_fn = activation_fn
        self.output_activation = output_activation

        self.layers = nn.ModuleList(list(map(lambda d: nn.Linear(*d), list(zip(dims, dims[1:])))))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers)-1 and self.output_activation is not None:
                x = self.output_activation(x)
            else:
                x = self.activation_fn(x)

        return x



class Encoder(nn.Module):
    def __init__(self, dims, sample_layer=GaussianSample):
        """
        Inference network

        Attempts to infer the probability distribution
        p(z|x) from the data by fitting a variational
        distribution q_phi(z|x). Returns the two parameters
        of the distribution (mu, log sig_sq).

        :param dims: dimensions of the networks
           given by the number of neurons on the form
           [input_dim, [hidden_dims], latent_dim].
        """
        super(Encoder, self).__init__()

        [x_dim, h_dim, z_dim] = dims
        neurons = [x_dim] + h_dim
        self.bn_cat = nn.BatchNorm1d(x_dim)
        self.drp_5 = nn.Dropout(.5)
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]
        bn_layers = [nn.BatchNorm1d(neurons[i]) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(linear_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        self.sample = sample_layer(h_dim[-1], z_dim)

    def forward(self, x):
        x = self.bn_cat(x)
        # x = self.drp_5(x)
        for i, (layer, bn_layer) in enumerate(zip(self.hidden, self.bn_layers)):
            x = F.relu(layer(x))
            #x = F.relu(self.drp_5(layer(x)))
	        #if i < len(self.hidden) - 2:
            #   x = F.relu(self.drp_5(bn_layer(layer(x))))
            #else:
            #    x = F.relu(self.drp_5(layer(x)))
        return self.sample(x)

class Decoder(nn.Module):
    def __init__(self, dims):
        """
        Generative network

        Generates samples from the original distribution
        p(x) by transforming a latent representation, e.g.
        by finding p_theta(x|z).

        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        super(Decoder, self).__init__()

        [z_dim, h_dim, x_dim] = dims
        neurons = [z_dim] + h_dim
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]
        self.hidden = nn.ModuleList(linear_layers)
        self.bn_cat = nn.BatchNorm1d(z_dim)
        bn_layers = [nn.BatchNorm1d(neurons[i]) for i in range(1, len(neurons))]
        self.reconstruction = nn.Linear(h_dim[-1], x_dim)
        self.bn_layers = nn.ModuleList(bn_layers)
        self.output_activation = nn.Sigmoid()
        self.drp_5 = nn.Dropout(.5)

    def forward(self, x):
        # x = self.bn_cat(x)
        # x = self.drp_5(x)
        for i, (layer, bn_layer) in enumerate(zip(self.hidden, self.bn_layers)):
            x = F.relu(layer(x))
            # if i == 0:
            #     x = F.relu(bn_layer(layer(x)))
            # else:
            #     x = F.relu(layer(x))
        return self.output_activation(self.reconstruction(x))


class VariationalAutoencoder(nn.Module):
    def __init__(self, dims):
        """
        Variational Autoencoder [Kingma 2013] model
        consisting of an encoder/decoder pair for which
        a variational distribution is fitted to the
        encoder. Also known as the M1 model in [Kingma 2014].

        :param dims: x, z and hidden dimensions of the networks
        """
        super(VariationalAutoencoder, self).__init__()

        [x_dim, z_dim, h_dim] = dims
        self.z_dim = z_dim
        self.flow = None

        self.encoder = Encoder([x_dim, h_dim, z_dim])
        self.decoder = Decoder([z_dim, list(reversed(h_dim)), x_dim])
        self.kl_divergence = 0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _kld(self, z, q_param, p_param=None):
        """
        Computes the KL-divergence of
        some element z.

        KL(q||p) = -∫ q(z) log [ p(z) / q(z) ]
                 = -E[log p(z) - log q(z)]

        :param z: sample from q-distribuion
        :param q_param: (mu, log_var) of the q-distribution
        :param p_param: (mu, log_var) of the p-distribution
        :return: KL(q||p)
        """
        (mu, log_var) = q_param

        if self.flow is not None:
            f_z, log_det_z = self.flow(z)
            qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
            z = f_z
        else:
            qz = log_gaussian(z, mu, log_var)

        if p_param is None:
            pz = log_standard_gaussian(z)
        else:
            (mu, log_var) = p_param
            pz = log_gaussian(z, mu, log_var)

        kl = qz - pz
        return kl

    def kl(self, z_mean, z_log_var):
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_log_var) + z_mean**2 - 1. - z_log_var, 1))
        if kl_loss < 0:
            import pdb;pdb.set_trace()
        return kl_loss

    def add_flow(self, flow):
        self.flow = flow

    def forward(self, x, y=None):
        """
        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters.

        :param x: input data
        :return: reconstructed input
        """
        z, z_mu, z_log_var = self.encoder(x)


        self.kl_divergence = self.kl(z_mu, z_log_var)
        #self.kl_divergence = self._kld(z, (z_mu, z_log_var))

        x_mu = self.decoder(z)

        return x_mu

    def sample(self, z):
        """
        Given z ~ N(0, I) generates a sample from
        the learned distribution based on p_θ(x|z).
        :param z: (torch.autograd.Variable) Random normal variable
        :return: (torch.autograd.Variable) generated sample
        """
        return self.decoder(z)



