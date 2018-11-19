from itertools import repeat

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import log_sum_exp, enumerate_discrete
from .distributions import log_standard_categorical
from layers import *
from precision_k import *

class ImportanceWeightedSampler(object):
    """
    Importance weighted sampler [Burda 2015] to
    be used in conjunction with SVI.
    """
    def __init__(self, mc=1, iw=1):
        """
        Initialise a new sampler.
        :param mc: number of Monte Carlo samples
        :param iw: number of Importance Weighted samples
        """
        self.mc = mc
        self.iw = iw

    def resample(self, x):
        return x.repeat(self.mc * self.iw, 1)

    def __call__(self, elbo):
        elbo = elbo.view(self.mc, self.iw, -1)
        elbo = torch.mean(log_sum_exp(elbo, dim=1, sum_op=torch.mean), dim=0)
        return elbo.view(-1)


class DeterministicWarmup(object):
    """
    Linear deterministic warm-up as described in
    [Sonderby 2016].
    """
    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1/n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t
        return self.t


class SVI(nn.Module):
    """
    Stochastic variational inference (SVI).
    """
    base_sampler = ImportanceWeightedSampler(mc=1, iw=1)
    def __init__(self, model, params, likelihood=F.binary_cross_entropy, beta=repeat(1), sampler=base_sampler):
        """
        Initialises a new SVI optimizer for semi-
        supervised learning.
        :param model: semi-supervised model to evaluate
        :param likelihood: p(x|y,z) for example BCE or MSE
        :param sampler: sampler for x and y, e.g. for Monte Carlo
        :param beta: warm-up/scaling of KL-term
        """
        super(SVI, self).__init__()
        self.model = model
        self.likelihood = likelihood
        self.sampler = sampler
        self.beta = beta
        self.params = params

    def forward(self, xs, ys=None, temp=None, normal=0):
        reconstruction = self.model(xs, ys)
        # diff = reconstruction - xs
        # likelihood = - torch.sum(torch.mul(diff, diff), dim=-1)
        # likelihood = - torch.sum(torch.abs(diff), dim=-1)
        likelihood = - torch.nn.functional.binary_cross_entropy(reconstruction, xs)*xs.shape[-1]
        prior = -log_standard_categorical(ys)
        
        xs = xs.data.cpu().numpy()
        reconstruction = reconstruction.data.cpu().numpy()
        P = precision_k(xs.astype(int), reconstruction,int(np.sum(xs, axis=1).mean()))
        L = likelihood + prior - self.params.reconFact * self.model.kl_divergence
        return - torch.mean(L) , np.mean(self.model.kl_divergence.data.cpu().numpy()), - np.mean(likelihood.data.cpu().numpy()), - np.mean(prior.data.cpu().numpy()), P