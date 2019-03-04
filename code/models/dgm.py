# -*- coding: utf-8 -*-
from sklearn import linear_model as lm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from networks import *
from layers import GaussianSample
from inference import log_gaussian, log_standard_gaussian

class DeepGenerativeModel(nn.Module):
    def __init__(self, params, sample_layer=GaussianSample):

        super(DeepGenerativeModel, self).__init__()
        self.params = params

        self.featureLearn = Encoder(params)
        self.decoder = Decoder(params)
        self.classifier = Classifier(params)
        self.sample = sample_layer(500, params.z_dim)
        #self.sample = sample_layer(128, params.z_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
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
        qz = log_gaussian(z, mu, log_var) 
        pz = log_standard_gaussian(z)

        kl = qz - pz
        return kl

    def forward(self, x, y):

        #hidden = self.featureLearn(x)
        hidden = self.featureLearn(torch.cat([x, y], dim=1))
        z, z_mu, z_log_var = self.sample(hidden)
        x_mu = self.decoder(torch.cat([z, y], dim=1))

        #self.kl_divergence = self.kl(z_mu, z_log_var)
        self.kl_divergence = self._kld(z, (z_mu, z_log_var))

        return x_mu

    def classify(self, x):

        #hidden = self.featureLearn(x)
        #logits = self.classifier(hidden)
        logProbs, preds = self.classifier(x)
        return logProbs, preds

    def kl(self, z_mean, z_log_var):

        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_log_var) + z_mean**2 - 1. - z_log_var, 1))
        return kl_loss

    def generate(self, cY):

        epsilon = torch.autograd.Variable(torch.randn((cY.shape[0], self.params.z_dim)), requires_grad=False).float()
        epsilon = epsilon.cuda() if cY.is_cuda else epsilon
        x_mu = self.decoder(torch.cat([epsilon, cY.float()], dim=1))

        return x_mu

    # def sample(self, z, y):

    #     y = y.float()
    #     x = self.decoder(torch.cat([z, y], dim=1))
    #     return x

