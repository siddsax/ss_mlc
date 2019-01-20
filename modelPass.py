import torch
import numpy as np
import sys
from functools import reduce
from operator import __or__
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from utils import *
from layers import *
from models import DeepGenerativeModel
from itertools import repeat, cycle
from torch.autograd import Variable
from inference import SVI, DeterministicWarmup, ImportanceWeightedSampler
from precision_k import *

def modelTrPass(model, optimizer, elbo, params, logFile, epoch, viz=None):

    model.train()
    m = len(params.unlabelled)

    for iterator, ((u, _), (x, y)) in enumerate(params.allData):

        params.kl_annealling = 1 - 1.0 * np.exp(- params.step*params.factor*1e-5)
        params.temperature = max(.5, 1.0*np.exp(-params.step*3e-4))

        x, y, u = Variable(x).squeeze().float(), Variable(y).squeeze().float(), Variable(u).squeeze().float()
        if params.cuda:
            x, y, u = x.cuda(device=0), y.cuda(device=0), u.cuda(device=0)

        _, preds = model.classify(x)
        classication_loss = params.alpha * torch.nn.functional.binary_cross_entropy(preds, y)*y.shape[-1]

        if params.ss:
            L, kl, recon, prior = elbo(x, y=y)
            U, klU, reconU, H, priorU = elbo(u, temperature=params.temperature, normal=params.normal)
            loss = L + classication_loss + U
        else:
            kl, klU, recon, reconU, H, prior, priorU  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            loss = classication_loss

        total_loss = loss.data.cpu().numpy()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        mseLoss = classication_loss.data.cpu().numpy()
        params.step += 1

        if(iterator % int(max(m/6, 5))==0):

          toPrint = "[TRAIN]:({}, {}/{});Total {:.2f}; KL_label {:.2f}, Recon_label {:.2f}; KL_ulabel {:.2f}, Recon_ulabel {:.2f}, entropy {:.2f}; Classify_loss {:.2f}; prior {:.2f}; priorU {:.2f}".format                     (float(params.epoch), float(iterator), float(m), float(total_loss), float(kl), float(recon), float(klU), float(reconU), float(H), float(classication_loss), float(prior), float(priorU                      ))
          print(toPrint)
          lossesT, losses_namesT = modelTePass(model, elbo, params, optimizer, logFile, testBatch=np.inf)

        mseLoss = mseLoss / params.alpha

    P = 100*precision_k(y.data.cpu().numpy().squeeze(),preds.data.cpu().numpy().squeeze(), 5)
    if params.ss:
        return [P[0], mseLoss, recon], ['Prec_1', 'BCELoss', 'lblLossTrain']
    else:
        return [P[0], mseLoss], ['Prec_1', 'BCELoss']

def modelTePass(model, elbo, params, optimizer, logFile, testBatch=5000):

  model.eval()
  mseLoss, total_loss, labelled_loss, unlabelled_loss, kl, recon, Lpred, Lgt, reconU  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  m = len(params.validation)
  ypred, ygt, XAll, dataPts = [], [], [], 0

  for x, y in params.validation:

      x, y = Variable(x).squeeze().float(), Variable(y).squeeze().float()
      dataPts +=x.shape[0]
      if dataPts > testBatch:
          break
      if params.cuda:
          x, y = x.cuda(device=0), y.cuda(device=0)

      U, _, reconAU, _, _ = elbo(x, temperature=params.temperature, normal=params.normal)
      L, klA, reconA, prior = elbo(x, y=y)
      logits, preds = model.classify(x)

      classication_loss = torch.nn.functional.binary_cross_entropy(preds, y)*y.shape[-1]
      loss = L + params.alpha * classication_loss + U

      total_loss += loss.data.cpu().numpy()
      labelled_loss += L.data.cpu().numpy()
      unlabelled_loss += U.data.cpu().numpy()
      mseLoss += classication_loss.data.cpu().numpy()
      kl += klA
      recon += reconA
      reconU += reconAU
      ypred.append(preds.data.cpu().numpy().squeeze())
      ygt.append(y.data.cpu().numpy().squeeze())
      XAll.append(x.data.cpu().numpy().squeeze())
      lp, _, _, _= elbo(x, y=gumbel_multiSample(logits, params.temperature))
      Lpred += lp.data.cpu().numpy()
      Lgt += L.data.cpu().numpy()

  ygt, ypred, XAll = np.concatenate(ygt, axis=0), np.concatenate(ypred, axis=0), np.concatenate(XAll, axis=0)
  P = 100*precision_k(ygt, ypred,5)

  if P[0] > params.bestP:
      params.bestP = P[0]

  toPrint = 'recon {:.2f}, reconU {:.2f} lblLossPred {:.2f}, lblLossGT {:.2f}'.format(float(recon/m), float(reconU/m), Lpred / m, Lgt/m)
  toPrint += " || Prec Best " + str(params.bestP) + " Prec. " + str(P[0])+ " " + str(P[2]) + " " + str(P[4])

  logFile.write(toPrint + '\n')
  logFile.flush()
  print(toPrint)
  optimizer.zero_grad()
  model.train()

  if params.ss:
        return [P[0], mseLoss / m, Lpred / m, Lgt/m], ['Prec_1_Test', 'BCELossTest', 'lblLossPred', 'lblLossGT']
  else:
        return [P[0], mseLoss / m], ['Prec_1_Test', 'BCELossTest']


