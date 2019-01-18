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

        #self.kl_divergence = self._kld(z, (z_mu, z_log_var))
        self.kl_divergence = self.kl(z_mu, z_log_var)
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
    def _construct_thresholds(self, probs, targets, top_k=None):

        nb_samples, nb_labels = targets.shape
        top_k = top_k or nb_labels

        # Sort predicted probabilities in descending order
        idx = np.argsort(probs, axis=1)[:,:-(top_k + 1):-1]
        p_sorted = np.vstack([probs[i, idx[i]] for i in range(len(idx))])
        t_sorted = np.vstack([targets[i, idx[i]] for i in range(len(idx))])

        # Compute F-1 measures for every possible threshold position
        F1 = []
        TP = np.zeros(nb_samples)
        FN = t_sorted.sum(axis=1)
        FP = np.zeros(nb_samples)
        for i in range(top_k):
            TP += t_sorted[:,i]
            FN -= t_sorted[:,i]
            FP += 1 - t_sorted[:,i]
            F1.append(2 * TP / (2 * TP + FN + FP))
        F1 = np.vstack(F1).T

        # Find the thresholds
        row = np.arange(nb_samples)
        col = F1.argmax(axis=1)
        p_sorted = np.hstack([p_sorted, np.zeros(nb_samples)[:, None]])
        T = 0.5 * (p_sorted[row, col] + p_sorted[row, col + 1])[:, None]

        return T

    def fit_thresholds(self, inputs, probs, Y, alpha=np.logspace(-3, 3, num=10).tolist(), cv=5, top_k=None):
      
        T = self._construct_thresholds(probs, Y)

        if isinstance(alpha, list):
            model = lm.RidgeCV(alphas=alpha, cv=cv).fit(inputs, T)
            alpha = model.alpha_
            self.t_models = lm.Ridge(alpha=alpha)
            self.t_models.fit(inputs, T)

    def predict_threshold(self, X, probs):
 
        T = self.t_models.predict(X)
        preds = probs >= T
        return preds

    def sample(self, z, y):

        y = y.float()
        x = self.decoder(torch.cat([z, y], dim=1))
        return x

