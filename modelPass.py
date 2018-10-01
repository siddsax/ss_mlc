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
from precision_k import *

def modelTrPass(model, optimizer, elbo, params, logFile, viz=None):
  model.train()
  iterator = 0
  m = len(params.unlabelled)
  # m = len(params.labelled)
  reconAl = 0.0
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
        # klU, reconU, H, priorU  = 0.0, 0.0, 0.0, 0.0
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

      if(iterator % int(max(m, 3))==0):
      # if(iterator>40 and iterator%2 == 0):#
      # if((iterator % 12)==0):
        toPrint = "[TRAIN]:({}, {}/{});Total {:.2f}; KL_label {:.2f}, Recon_label {:.2f}; KL_ulabel {:.2f}, Recon_ulabel {:.2f}, entropy {:.2f}; Classify_loss {:.2f}; prior {:.2f}; priorU {:.2f}".format(
          float(params.epoch), float(iterator), float(m), float(total_loss), float(kl), float(reconAl/m), float(klU), float(reconU), float(H), float(classication_loss), float(prior), float(priorU)
        )
        print(toPrint)
      	reconAl = 0.0
        model.fit_thresholds(x.data.cpu().numpy(), preds.data.cpu().numpy(), y.data.cpu().numpy())
        ############## NOT USING ALL DATA ###############################################################
        lossesT, losses_namesT = modelTePass(model, elbo, params, optimizer, logFile)#, testBatch=np.inf)
        #################################################################################################
      mseLoss = mseLoss / params.alpha
      reconAl += recon
  P = 100*precision_k(y.data.cpu().numpy().squeeze(),preds.data.cpu().numpy().squeeze(), 5)
  if params.ss:
    return [P[0], mseLoss, 100*params.temp, recon], ['Prec_1', 'BCELoss', 'Temperaturex100', 'lblLossTrain']
  else:
    return [P[0], mseLoss], ['Prec_1', 'BCELoss']

def modelTePass(model, elbo, params, optimizer, logFile, testBatch=5000):
  model.eval()
  mseLoss, total_loss, labelled_loss, unlabelled_loss, kl, recon, Lpred, Lgt, reconU  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  m = len(params.validation)
  dataPts, XAll, ygt, ypred = 0, [], [], []
  recon2 = 0.0
  for x, y in params.validation:
      x, y = Variable(x).squeeze().float(), Variable(y).squeeze().float()
      dataPts +=x.shape[0]
      if dataPts > testBatch:
        break
      if params.cuda:
          x, y = x.cuda(device=0), y.cuda(device=0)

      U, _, reconAU, _, _ = elbo(x, temp=params.temp, normal=params.normal)
      L, klA, reconA, prior = elbo(x, y=y)
      logits, preds = model.classify(x)

      classication_loss = torch.nn.functional.binary_cross_entropy(preds, y)*y.shape[-1]
      J_alpha = L + params.alpha * classication_loss + U

      total_loss += J_alpha.data.cpu().numpy()
      labelled_loss += L.data.cpu().numpy()
      unlabelled_loss += U.data.cpu().numpy()
      mseLoss += classication_loss.data.cpu().numpy()#torch.mean((pred_idx.data == lab_idx.data).float())
      kl += klA
      recon += reconA
      reconU += reconAU
      ypred.append(preds.data.cpu().numpy().squeeze())
      ygt.append(y.data.cpu().numpy().squeeze())
      XAll.append(x.data.cpu().numpy().squeeze())
      lp, _, _, _= elbo(x, y=gumbel_multiSample(logits, params.temp))
      Lpred += lp.data.cpu().numpy()
      Lgt += L.data.cpu().numpy()

      reconstruction = model.generate(y)
      diff = reconstruction - x
      recon2 += torch.sum(torch.mul(diff, diff), dim=-1).data.cpu().numpy().mean()

  ygt, ypred, XAll = np.concatenate(ygt, axis=0), np.concatenate(ypred, axis=0), np.concatenate(XAll, axis=0)
  P = 100*precision_k(ygt, ypred,5)
  if params.epoch:
    if P[0] > params.bestP:
      params.bestP = P[0]
      save_model(model, optimizer, params.epoch, params, "/model_best_class_" + params.mn + "_" + str(params.ss))

    if float(recon2/m) < params.bestR:
      params.bestR = float(recon2/m)
      save_model(model, optimizer, params.epoch, params, "/model_best_regen_" + params.mn + "_" + str(params.ss))

  toPrint = 'recon2 {:.6} recon {:.6f}, reconU {:.6f} lblLossPred {:.2f}, lblLossGT {:.2f} best recon2 {:.6f}'.format(float(recon2/m), float(recon/m), float(reconU/m), Lpred / m, Lgt/m, float(params.bestP))
  toPrint += " || Prec Best " + str(params.bestP) + " Prec. " + str(P[0])+ " " + str(P[2]) + " " + str(P[4])
  print("-"*20)

  if params.epoch:
    preds = model.predict_threshold(XAll, ypred)
    f1_macro = f1_measure(ygt, preds, average='macro')
    f1_micro = f1_measure(ygt, preds, average='micro')
    toPrint += "f1_macro {:.2f} f1_micro {:.2f}".format(100*f1_macro, 100*f1_micro)

  logFile.write(toPrint + '\n')
  logFile.flush()
  print(toPrint)
  optimizer.zero_grad()
  model.train()
  if params.ss:
        return [P[0], mseLoss / m, Lpred / m, Lgt/m], ['Prec_1_Test', 'BCELossTest', 'lblLossPred', 'lblLossGT']
  else:
        return [P[0], mseLoss / m], ['Prec_1_Test', 'BCELossTest']
  # return [P[0], mseLoss / m], ['Prec_1_Test', 'BCELossTest']
