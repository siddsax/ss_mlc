# -*- coding: utf-8 -*-
from sklearn import linear_model as lm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from .vae import VariationalAutoencoder
from .vae import Encoder, Decoder

def weights_init(m):
    if(torch.__version__=='0.4.0'):
    	torch.nn.init.xavier_uniform_(m)
    else:
	torch.nn.init.xavier_uniform(m)

class DeepGenerativeModel(VariationalAutoencoder):
    def __init__(self, dims, params):
        """
        M2 code replication from the paper
        'Semi-Supervised Learning with Deep Generative Models'
        (Kingma 2014) in PyTorch.

        The "Generative semi-supervised model" is a probabilistic
        model that incorporates label information in both
        inference and generation.

        Initialise a new generative model
        :param dims: dimensions of x, y, z and hidden layers.
        """
        [x_dim, self.y_dim, z_dim, h_dim] = dims
        super(DeepGenerativeModel, self).__init__([x_dim, z_dim, h_dim])
        self.z_dim = z_dim
        self.encoder = Encoder([x_dim + self.y_dim, h_dim, z_dim])
        self.decoder = Decoder([z_dim + self.y_dim, list(reversed(h_dim)), x_dim])
        # self.gumbel = GumbelSoftmax(h_dim[0], self.y_dim, 10)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):
        # Add label and data and generate latent variable
        z, z_mu, z_log_var = self.encoder(torch.cat([x, y], dim=1))

        self.kl_divergence = self._kld(z, (z_mu, z_log_var))

        # Reconstruct data point from latent data and label
        x_mu = self.decoder(torch.cat([z, y], dim=1))

        return x_mu

    def generate(self, cY):

        epsilon = torch.autograd.Variable(torch.randn((cY.shape[0], self.z_dim)), requires_grad=False).float()
        if cY.is_cuda:
            epsilon = epsilon.cuda()
        x_mu = self.decoder(torch.cat([epsilon, cY.float()], dim=1))
        return x_mu

    def sample(self, z, y):
        """
        Samples from the Decoder to generate an x.
        :param z: latent normal variable
        :param y: label (one-hot encoded)
        :return: x
        """
        y = y.float()
        x = self.decoder(torch.cat([z, y], dim=1))
        return x
