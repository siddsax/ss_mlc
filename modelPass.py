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
from models import DeepGenerativeModel
from itertools import repeat, cycle
from torch.autograd import Variable
from inference import SVI, DeterministicWarmup, ImportanceWeightedSampler
from precision_k import precision_k

def modelTrPass(model, optimizer, elbo, params):
  model.train()
  print("Epoch: {}".format(params.epoch))
  total_loss, labelled_loss, unlabelled_loss, accuracy = (0, 0, 0, 0)
 
  for (x, y), (u, _) in zip(cycle(params.labelled), params.unlabelled):
      params.temp = max(0.3, np.exp(-params.step*1e-4)) 
      x, y, u = Variable(x).squeeze(), Variable(y).squeeze(), Variable(u).squeeze()
      if params.cuda:
        x, y, u = x.cuda(device=0), y.cuda(device=0), u.cuda(device=0)

      L = -elbo(x, y=y)
      U = -elbo(u, temp=params.temp, normal=params.normal)

      # Add auxiliary classification loss q(y|x)
      logits = model.classify(x)
      classication_loss = torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

      J_alpha = - params.alpha * classication_loss
      if params.ss:
        J_alpha += L + U

      J_alpha.backward()
      optimizer.step()
      optimizer.zero_grad()

      total_loss += J_alpha.data[0]
      labelled_loss += L.data[0]
      unlabelled_loss += U.data[0]

      _, pred_idx = torch.max(logits, 1)
      _, lab_idx = torch.max(y, 1)
      accuracy += torch.mean((pred_idx.data == lab_idx.data).float())
      params.step += 1

  m = len(params.unlabelled)
  print("[TRAIN]:Total Loss {}, Labelled Loss {}, unlabelled loss {}, acc {}, temperature {}".format(
      total_loss / m, labelled_loss / m, unlabelled_loss / m, accuracy / m, params.temp))

def modelTePass(model, elbo, params):
  model.eval()

  total_loss, labelled_loss, unlabelled_loss, accuracy = (0, 0, 0, 0)
  m = len(params.validation)
  ypred = []
  ygt = []
  for x, y in params.validation:
      x, y = Variable(x).squeeze(), Variable(y).squeeze()
      if params.cuda:
          x, y = x.cuda(device=0), y.cuda(device=0)

      U = -elbo(x, temp=params.temp, normal=params.normal)
      L = -elbo(x, y=y)
      logits = model.classify(x)
      classication_loss = -torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

      J_alpha = L + params.alpha * classication_loss + U

      total_loss += J_alpha.data[0]
      labelled_loss += L.data[0]
      unlabelled_loss += U.data[0]
      _, pred_idx = torch.max(logits, 1)
      _, lab_idx = torch.max(y, 1)
      accuracy += torch.mean((pred_idx.data == lab_idx.data).float())
      ypred.append(logits.data.cpu().numpy().squeeze())
      ygt.append(y.data.cpu().numpy().squeeze())

  if accuracy / m > params.best:
    params.best = accuracy / m
  P = 100*precision_k(np.concatenate(ygt, axis=0), np.concatenate(ypred, axis=0),5)
  print("[TEST]:Total Loss {}, Labelled Loss {}, unlabelled loss {}, acc {}, best acc. {}".format(
      total_loss / m, labelled_loss / m, unlabelled_loss / m, accuracy / m, params.best))
from precision_k import precision_k
