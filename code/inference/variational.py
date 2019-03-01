from itertools import repeat

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import log_sum_exp, enumerate_discrete
from .distributions import log_standard_categorical
from layers import *
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
    def __init__(self, model, params, recon_loss=F.binary_cross_entropy, beta=repeat(1), sampler=base_sampler):
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
        self.recon_loss = recon_loss
        self.sampler = sampler
        self.beta = beta
        self.params = params

    def forward(self, x, y=None, temperature=None, normal=0):
        is_labelled = False if y is None else True

        #if not is_labelled:
        #    x = x.repeat(5, 1)

        # Enumerate choices of label
        logits, preds = self.model.classify(x)
        if not is_labelled:
            if normal:
                y = enumerate_discrete(x, self.params.y_dim)
                x = x.repeat(self.params.y_dim, 1)
              
            else:
                if temperature is None:
                    print("Error, temperatureerature not given: Exiting")
                    exit()

                x = x.repeat(5, 1)
                preds = preds.repeat(5, 1)
                logits = logits.repeat(5, 1, 1)

                #y = gumbel_softmax(preds, temperature)
                y = gumbel_multiSample(logits, temperature)
               

        reconstruction = self.model(x, y)

        # p(x|y,z)
        # diff = reconstruction - x
        recon_loss = self.recon_loss(reconstruction, x)#torch.sum(torch.mul(diff, diff), dim=-1)
       
        #recon_loss = - torch.sum(torch.mul(x, torch.log(reconstruction + 1e-5)) + torch.mul(1 - x, torch.log(1 - reconstruction + 1e-5)), dim=-1)
        #if torch.mean(recon_loss) < 0:
        #    import pdb;pdb.set_trace()
        
        # p(y)
        prior = log_standard_categorical(y)

        L = recon_loss + prior + self.model.kl_divergence #* float(self.params.kl_annealling)

        if is_labelled:
            return torch.mean(L) , np.mean(self.model.kl_divergence.data.cpu().numpy()), np.mean(recon_loss.data.cpu().numpy()), np.mean(prior.data.cpu().numpy())

        if normal:
            L = L.view_as(preds.t()).t()
            L = torch.sum(torch.mul(preds, L), dim=-1)

        #H = - (torch.sum(torch.mul(preds, torch.log(preds + 1e-8)) + torch.mul(1 - preds, torch.log(1 - preds + 1e-8)), dim=-1))
        H = torch.sum(torch.mul(preds, torch.log(preds + 1e-8)), dim=-1)
        #import pdb;pdb.set_trace()
        # Carefully written
        U = L + H #* float(self.model.kl_divergence)

        return torch.mean(U) , np.mean(self.model.kl_divergence.data.cpu().numpy()), np.mean(recon_loss.data.cpu().numpy()), np.mean(H.data.cpu().numpy()), np.mean(prior.data.cpu().numpy())
