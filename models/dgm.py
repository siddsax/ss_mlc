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

class Classifier(nn.Module):
    def __init__(self, dims, twoOut):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(Classifier, self).__init__()
        [x_dim, h_dim, y_dim] = dims
        self.twoOut = twoOut
        self.drp_5 = nn.Dropout(.5)
        self.dense = nn.Linear(x_dim, int(1.5*h_dim))
        self.dense_2 = nn.Linear(int(1.5*h_dim), h_dim)
        self.logits = nn.Linear(h_dim, y_dim)
        self.logitsP = nn.Linear(h_dim, y_dim)
        self.logitsN = nn.Linear(h_dim, y_dim)

        self.bn = nn.BatchNorm1d(h_dim)

    def forward(self, x):
        # x = self.drp_5(x)
        x = self.dense(x)
        x = F.relu(x)

        x = self.drp_5(x)
        x = self.dense_2(x)
        x = F.relu(x)
        #------------------------------------------------------
        if self.twoOut:
            predsP = self.logitsP(x)
            predsN = self.logitsN(x)
            predsP = predsP.view(predsP.shape[0], predsP.shape[1], 1)
            predsN = predsN.view(predsN.shape[0], predsN.shape[1], 1)
            logits = torch.cat((predsP, predsN), dim=-1)
            preds = F.softmax(logits, dim=-1)[:,:,0]
            return logits, preds
        ########################################################
        else:
            x = F.sigmoid(self.logits(x))
            x1 = x.view(x.shape[0], x.shape[1], 1)
            logits = torch.cat((x1, 1 - x1), dim=-1)
            return torch.log(logits+1e-8), x
        ########################################################


class DeepGenerativeModel(VariationalAutoencoder):
    def __init__(self, dims, params):

        [x_dim, self.y_dim, self.z_dim, h_dim] = dims

        super(DeepGenerativeModel, self).__init__([x_dim, self.z_dim, h_dim])

        self.encoder = Encoder([x_dim + self.y_dim, h_dim, self.z_dim])
        self.decoder = Decoder([self.z_dim + self.y_dim, list(reversed(h_dim)), x_dim])
        self.classifier = Classifier([x_dim, 600, self.y_dim], params.twoOut)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):

        z, z_mu, z_log_var = self.encoder(torch.cat([x, y], dim=1))
        x_mu = self.decoder(torch.cat([z, y], dim=1))

        self.kl_divergence = self._kld(z, (z_mu, z_log_var))

        return x_mu

    def classify(self, x):
        
        logits = self.classifier(x)
        return logits

    def generate(self, cY):

        epsilon = torch.autograd.Variable(torch.randn((cY.shape[0], self.z_dim)), requires_grad=False).float()
        if cY.is_cuda:
            epsilon = epsilon.cuda()
        x_mu = self.decoder(torch.cat([epsilon, cY.float()], dim=1))
        return x_mu

    def sample(self, z, y):

        y = y.float()
        x = self.decoder(torch.cat([z, y], dim=1))
        return x
