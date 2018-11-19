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

def modelTrPass(model, optimizer, elbo, params, logFile, viz=None):
  model.train()
  iterator = 0
  m = len(params.dataTr)
  reconAl = 0.0
  for (x,y) in params.dataTr:
      iterator += 1.0
      params.reconFact = float(1 - np.exp(-params.step*params.factor*1e-5))
      params.temp = max(.5, 1.0*np.exp(-params.step*3e-4))

      x, y = Variable(x).squeeze().float(), Variable(y).squeeze().float()
      if params.cuda:
        x, y = x.cuda(device=0), y.cuda(device=0)

      J_alpha, kl, recon, prior, _ = elbo(x, ys=y)
      total_loss = J_alpha.data.cpu().numpy()
      J_alpha.backward()
      optimizer.step()
      optimizer.zero_grad()
      params.step += 1

      if(iterator % int(max(m, 3))==0):

        toPrint = "[TRAIN]:({}, {}/{});Total {:.2f}; KL_label {:.2f}, Recon_label {:.2f}; prior {:.2f}".format(
          float(params.epoch), float(iterator), float(m), float(total_loss), float(kl), float(reconAl/m), float(prior))
        print(toPrint)
      	reconAl = 0.0
        ############## NOT USING ALL DATA ###############################################################
        # lossesT, losses_namesT = modelTePass(model, elbo, params, optimizer, logFile)#, testBatch=np.inf)
        #################################################################################################
      reconAl += recon
  return [100*params.reconFact, recon], ['KLFactor', 'reconXTrain']

def modelTePass(model, elbo, params, optimizer, logFile, testBatch=5000):
  model.eval()
  total_loss, kl, recon, PX, dataPts, XAll, reconAll = 0.0, 0.0, 0.0, 0.0, 0, [], []
  m = len(params.dataTe)
  reconZ = 0.0
  for x, y in params.dataTe:
      x, y = Variable(x).squeeze().float(), Variable(y).squeeze().float()
      dataPts +=x.shape[0]
      if dataPts > testBatch:
        break
      if params.cuda:
          x, y = x.cuda(device=0), y.cuda(device=0)

      J_alpha, klA, reconA, _, pX = elbo(x, ys=y)
      total_loss += J_alpha.data.cpu().numpy()
      kl += klA
      recon += reconA
      PX += pX
      reconstruction = model.generate(y)
      XAll.append(x.data.cpu().numpy().squeeze())
      reconAll.append(reconstruction.data.cpu().numpy().squeeze())
      diff = reconstruction - x
      reconZ += torch.sum(torch.mul(diff, diff), dim=-1).data.cpu().numpy().mean()

  XAll = np.concatenate(XAll, axis=0)
  reconAll = np.concatenate(reconAll, axis=0)
  k = int(np.sum(XAll, axis=1).mean())
  PZ = precision_k(XAll.astype(int), reconAll,k)
  PRand = precision_k(XAll.astype(int), np.random.rand(reconAll.shape[0], reconAll.shape[1]),k)
  
  if params.epoch:
    if float(reconZ/m) < params.bestRZ:
      params.bestRZ = float(reconZ/m)
      save_model(model, optimizer, params.epoch, params, "/model_best_1_" + params.mn )
    if float(recon/m) < params.bestRX:
      params.bestRX = float(recon/m)
      save_model(model, optimizer, params.epoch, params, "/model_best_2_" + params.mn )
    if float(PZ) > params.bestPZ:
      params.bestPZ = float(PZ)
      save_model(model, optimizer, params.epoch, params, "/model_best_3_" + params.mn )
    if float(PX/m) > params.bestPX:
      params.bestPX = float(PX/m)
      save_model(model, optimizer, params.epoch, params, "/model_best_4_" + params.mn )
  else:
      params.bestPZ = -1.0
  toPrint = 'reconZ {:.6} reconX {:.6f}, PrecisionZ@{} {:.6f} PrecisionX@{} {:.6f} PrecisionRand {:.6}, BestPZ  {:.6}'.format(\
  float(reconZ/m), float(recon/m), k, float(PZ), k, float(PX/m), float(PRand), float(params.bestPZ))

  logFile.write(toPrint + '\n')
  logFile.flush()
  print(toPrint)
  optimizer.zero_grad()
  model.train()
  return [float(reconZ/m), float(recon/m), PZ, float(PX/m)], ['reconZ', 'reconX', 'PrecisionZ@' + str(k), 'PrecisionX@' + str(k)]
