import torch
import numpy as np
import sys
from functools import reduce
from operator import __or__
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
# sys.path.append("../semi-supervised")
from utils import *
from layers import *
from models import DeepGenerativeModel
from itertools import repeat, cycle
from torch.autograd import Variable
from inference import SVI, DeterministicWarmup, ImportanceWeightedSampler
from precision_k import precision_k

def modelTrPass(model, optimizer, elbo, params):
  model.train()
  print("Epoch: {}".format(params.epoch))
  total_loss, labelled_loss, unlabelled_loss, mseLoss = (0, 0, 0, 0)
  iterator = 0
  m = len(params.unlabelled)
  for (x, y), (u, _) in zip(cycle(params.labelled), params.unlabelled):
     
      
      iterator += 1.0
      params.temp = max(0.3, np.exp(-params.step*1e-4))
      x, y, u = Variable(x).squeeze(), Variable(y).squeeze(), Variable(u).squeeze()
      if params.cuda:
            x, y, u = x.cuda(device=0), y.cuda(device=0), u.cuda(device=0)

      L, kl, recon = elbo(x, y=y)
      U, _, _ = elbo(u, temp=params.temp, normal=params.normal)

      # Add auxiliary classification loss q(y|x)
      logits = model.classify(x)
      classication_loss = - torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()
    #   classication_loss = torch.nn.functional.binary_cross_entropy(logits, y)*y.shape[-1]

      J_alpha = params.alpha * classication_loss
      if params.ss:
            J_alpha += L + U

      J_alpha.backward()
      optimizer.step()
      optimizer.zero_grad()

      total_loss = J_alpha.data[0]
      labelled_loss = L.data[0]
      unlabelled_loss = U.data[0]

      _, pred_idx = torch.max(logits, 1)
      _, lab_idx = torch.max(y, 1)
      mseLoss = classication_loss.data.cpu().numpy()#torch.mean((pred_idx.data == lab_idx.data).float())
      params.step += 1
      P = 100*precision_k(y.data.cpu().numpy().squeeze(),logits.data.cpu().numpy().squeeze(), 5)
      if(iterator % int(max(m/12, 5))==0):
        toPrint = "[TRAIN]:Total Loss {:.2f}, Labelled Loss {:.2f}, KL {:.2f}, recon {:.2f}, unlabelled loss {:.2f}, mseLoss {:.2f}, temperature {}".format(
            float(total_loss), float(labelled_loss), float(kl), float(recon), float(unlabelled_loss), float(mseLoss), params.temp)
        toPrint += " || Prec. " + str(P[0])
        # for i in range(5):
        #     toPrint += "{} ".format(P[i])
        print(toPrint)
        # modelTePass(model, elbo, params)
  return [P[0], mseLoss], ['Prec_1', 'BCELoss']

def modelTePass(model, elbo, params, optimizer):
  model.eval()

  total_loss, labelled_loss, unlabelled_loss, mseLoss = (0, 0, 0, 0)
  m = len(params.validation)
  ypred = []
  ygt = []
  Lgt = 0.0
  Lpred = 0.0
  for x, y in params.validation:
    
   
      x, y = Variable(x).squeeze(), Variable(y).squeeze()
      if params.cuda:
          x, y = x.cuda(device=0), y.cuda(device=0)

      U, _, _ = elbo(x, temp=params.temp, normal=params.normal)
      L, kl, recon = elbo(x, y=y)
      logits = model.classify(x)

    #   classication_loss = torch.nn.functional.binary_cross_entropy(logits, y)*y.shape[-1]
      classication_loss = -torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()
      J_alpha = L + params.alpha * classication_loss + U

      total_loss += J_alpha.data.cpu().numpy()
      labelled_loss += L.data.cpu().numpy()
      unlabelled_loss += U.data.cpu().numpy()
      _, pred_idx = torch.max(logits, 1)
      _, lab_idx = torch.max(y, 1)
      mseLoss += classication_loss.data.cpu().numpy()#torch.mean((pred_idx.data == lab_idx.data).float())
      ypred.append(logits.data.cpu().numpy().squeeze())
      ygt.append(y.data.cpu().numpy().squeeze())

      lp, _, _= elbo(x, y=gumbel_multiSample(logits, params.temp))
      Lpred += lp.data.cpu().numpy()
      Lgt += L.data.cpu().numpy()

  P = 100*precision_k(np.concatenate(ygt, axis=0), np.concatenate(ypred, axis=0),5)
  if P[0] > params.bestP:
    params.bestP = P[0]
  if mseLoss / m < params.best:
    params.best = mseLoss / m
    save_model(model, optimizer, params.epoch, params, "/model_best_test")
  toPrint = "[TEST]:Total Loss {:.2f}, Labelled Loss {:.2f}, KL {:.2f}, recon {:.2f}, unlabelled loss {:.2f}, mseLoss {:.2f}, best_p1 {}".format(
        float(total_loss), float(labelled_loss), float(kl), float(recon), float(unlabelled_loss), float(mseLoss), params.bestP)
  toPrint += " || Prec. " + str(P[0])
#   for i in range(5):
#       toPrint += "{} ".format(P[i])
  print(toPrint)
  print("="*100)
  return [P[0], mseLoss / m, Lpred / m, Lgt/m], ['Prec_1_Test', 'BCELossTest', 'lblLossPred', 'lblLossGT']

