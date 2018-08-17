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

def modelTrPass(model, optimizer, elbo, params):
  model.train()
  print("Epoch: {}".format(params.epoch))
  total_loss, labelled_loss, unlabelled_loss, accuracy = (0, 0, 0, 0)
  
  for (x, y), (u, _) in zip(cycle(params.labelled), params.unlabelled):
      x, y, u = Variable(x), Variable(y), Variable(u)
      if params.cuda:
        x, y, u = x.cuda(device=0), y.cuda(device=0), u.cuda(device=0)

      L = -elbo(x, y)
      U = -elbo(u)

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

  m = len(params.unlabelled)
  print("[TRAIN]:Total Loss {}, Labelled Loss {}, unlabelled loss {}, acc {}".format(total_loss / m, labelled_loss / m, unlabelled_loss / m, accuracy / m))

def modelTePass(model, elbo, params):
  model.eval()

  total_loss, labelled_loss, unlabelled_loss, accuracy = (0, 0, 0, 0)
  for x, y in params.validation:
      x, y = Variable(x), Variable(y)

      if params.cuda:
          x, y = x.cuda(device=0), y.cuda(device=0)

      L = -elbo(x, y)
      U = -elbo(x)

      logits = model.classify(x)
      classication_loss = -torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

      J_alpha = L + params.alpha * classication_loss + U

      total_loss += J_alpha.data[0]
      labelled_loss += L.data[0]
      unlabelled_loss += U.data[0]

      _, pred_idx = torch.max(logits, 1)
      _, lab_idx = torch.max(y, 1)
      accuracy += torch.mean((pred_idx.data == lab_idx.data).float())

  m = len(params.validation)
  if accuracy / m > params.best:
    params.best = accuracy / m
  print("[TEST]:Total Loss {}, Labelled Loss {}, unlabelled loss {}, acc {}, best acc. {}".format(
      total_loss / m, labelled_loss / m, unlabelled_loss / m, accuracy / m, params.best))
