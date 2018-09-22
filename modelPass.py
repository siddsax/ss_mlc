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

def modelTrPass(model, optimizer, elbo, params, viz=None):
  model.train()
  iterator = 0
  m = len(params.unlabelled)
  # m = len(params.labelled)
  for (u, _), (x, y) in params.allData:
  # for (x,y) in params.unlabelled:
      iterator += 1.0
      # np.exp(-params.step*3e-4)
      params.reconFact = torch.autograd.Variable(torch.from_numpy(np.array(1 - np.exp(-params.step*params.factor*1e-5)))).float()
      if torch.cuda.is_available():
        params.reconFact = params.reconFact.cuda()
      params.temp = max(.5, 1.0*np.exp(-params.step*3e-4)) #default
      # params.temp = .5 + .5* np.exp(-params.step*2.7e-4)

      x, y = Variable(x).squeeze().float(), Variable(y).squeeze().float()
      if params.cuda:
        x, y = x.cuda(device=0), y.cuda(device=0)

      # Add auxiliary classification loss q(y|x)
      _, preds = model.classify(x)
      # classication_loss = - torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()
      classication_loss = params.alpha * torch.nn.functional.binary_cross_entropy(preds, y)*y.shape[-1]


      if params.ss:
        L, kl, recon, prior = elbo(x, y=y)
        u = Variable(u).squeeze().float()
        if params.cuda:
          u = u.cuda(device=0)
        U, klU, reconU, H, priorU = elbo(u, temp=params.temp, normal=params.normal)
        J_alpha = L + classication_loss + U
      else:
        kl, klU, recon, reconU, H, prior, priorU  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        recon = 0.0
        J_alpha = classication_loss

      total_loss = J_alpha.data.cpu().numpy()
      J_alpha.backward()
      optimizer.step()
      optimizer.zero_grad()

      mseLoss = classication_loss.data.cpu().numpy()
      params.step += 1

      if(iterator % int(max(m/12, 5))==0):
      # if((iterator % 12)==0):
        toPrint = "[TRAIN]:({}, {}/{});Total {:.2f}; KL_label {:.2f}, Recon_label {:.2f}; KL_ulabel {:.2f}, Recon_ulabel {:.2f}, entropy {:.2f}; Classify_loss {:.2f}; prior {:.2f}; priorU {:.2f}".format(
          float(params.epoch), float(iterator), float(m), float(total_loss), float(kl), float(recon), float(klU), float(reconU), float(H), float(classication_loss), float(prior), float(priorU)
        )
        print(toPrint)
      mseLoss = mseLoss / params.alpha

  P = 100*precision_k(y.data.cpu().numpy().squeeze(),preds.data.cpu().numpy().squeeze(), 5)
  if params.ss:
    return [P[0], mseLoss, 100*params.temp, recon], ['Prec_1', 'BCELoss', 'Temperaturex100', 'lblLossTrain']
  else:
    return [P[0], mseLoss], ['Prec_1', 'BCELoss']

def modelTePass(model, elbo, params, optimizer, testBatch=5000):
  model.eval()
  total_loss, labelled_loss, unlabelled_loss, mseLoss = (0, 0, 0, 0)
  m = len(params.validation)
  ypred = []
  ygt = []
  Lgt = 0.0
  Lpred = 0.0
  kl = 0.0
  recon = 0.0
  dataPts = 0
  for x, y in params.validation:
      x, y = Variable(x).squeeze().float(), Variable(y).squeeze().float()
      dataPts +=x.shape[0]
      if dataPts > testBatch:
        break
      if params.cuda:
          x, y = x.cuda(device=0), y.cuda(device=0)

      U, _, _, _, _ = elbo(x, temp=params.temp, normal=params.normal)
      L, klA, reconA, prior = elbo(x, y=y)
      logits, preds = model.classify(x)

      # classication_loss = -torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()
      classication_loss = torch.nn.functional.binary_cross_entropy(preds, y)*y.shape[-1]
      J_alpha = L + params.alpha * classication_loss + U

      total_loss += J_alpha.data.cpu().numpy()
      labelled_loss += L.data.cpu().numpy()
      unlabelled_loss += U.data.cpu().numpy()
      mseLoss += classication_loss.data.cpu().numpy()#torch.mean((pred_idx.data == lab_idx.data).float())
      kl += klA
      recon += reconA
      ypred.append(preds.data.cpu().numpy().squeeze())
      ygt.append(y.data.cpu().numpy().squeeze())

      lp, _, _, _= elbo(x, y=gumbel_multiSample(logits, params.temp))
      Lpred += lp.data.cpu().numpy()
      Lgt += L.data.cpu().numpy()

  P = 100*precision_k(np.concatenate(ygt, axis=0), np.concatenate(ypred, axis=0),5)
  if P[0] > params.bestP:
    params.bestP = P[0]
  if mseLoss / m < params.best:
    params.best = mseLoss / m
    save_model(model, optimizer, params.epoch, params, "/model_best_test_" + params.mn + "_" + str(params.ss))
  toPrint = "[TEST]:Temp {:.3f}, Factor {:.3f}, Total Loss {:.2f}, Labelled Loss {:.2f}, KL {:.2f}, recon {:.2f}, unlabelled loss {:.2f}, mseLoss {:.2f}, best_p1 {}, best_bce {:.2f}".format(
        float(params.temp), params.reconFact.data.cpu().numpy(), float(total_loss / m), float(labelled_loss/ m), float(kl/m), float(recon/m), float(unlabelled_loss/ m), float(mseLoss/ m), params.bestP, params.best)
  # toPrint += " || Prec. " + str(P[0])+ " " + str(P[2]) + " " + str(P[4])
  toPrint = "Prec Best " + str(params.bestP) + " Prec. " + str(P[0])+ " " + str(P[2]) + " " + str(P[4])

  print(toPrint)
  print("="*100)
  if params.ss:
    return [P[0], mseLoss / m, Lpred / m, Lgt/m], ['Prec_1_Test', 'BCELossTest', 'lblLossPred', 'lblLossGT']
  else:
    return [P[0], mseLoss / m], ['Prec_1_Test', 'BCELossTest']
