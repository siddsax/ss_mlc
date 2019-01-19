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

def modelTrPass(model, optimizer, elbo, params, logFile, epoch, viz=None):

    model.train()

    m = len(params.unlabelled)

    for iterator, ((u, _), (x, y)) in enumerate(zip(params.unlabelled, params.labelled)):

        params.kl_annealling = 1 - 1.0 * np.exp(- params.step*params.factor*1e-5)
        params.temperature = max(.5, 1.0 * np.exp(- params.step*3e-4))

        x, y, u = Variable(x).squeeze().float(), Variable(y).squeeze().float(), Variable(u).squeeze().float()
        if params.cuda:
            x, y, u = x.cuda(device=0), y.cuda(device=0), u.cuda(device=0)

        # Add auxiliary classification loss q(y|x)
        logits, preds = model.classify(x)
        # classication_loss = - torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()
        classication_loss = params.alpha * torch.nn.functional.binary_cross_entropy(preds, y)*y.shape[-1]

        if params.ss:
            L, kl, recon, prior = elbo(x, y=y)
            U, klU, reconU, H, priorU = elbo(u, temperature=params.temperature, normal=params.normal)
            loss = L + classication_loss + U
        else:
            kl, klU, recon, reconU, H, prior, priorU  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            recon = 0.0
            loss = classication_loss

        total_loss = loss.data.cpu().numpy()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        classication_loss = classication_loss.data.cpu().numpy()
        params.step += 1

        if(iterator % int(max(len(params.unlabelled)/3, 3))==0):
            toPrint = "[TRAIN]:({}, {}/{});Total {:.2f}; KL_label {:.2f}, Recon_label {:.2f}; KL_ulabel {:.2f}, Recon_ulabel {:.2f}, \
            entropy {:.2f}; Classify_loss {:.2f}; prior {:.2f}; priorU {:.2f}".format(float(params.epoch), float(iterator), \
            float(len(params.unlabelled)), float(loss.data.cpu().numpy()), float(kl), float(recon), float(klU), float(reconU),\
            float(H), float(classication_loss), float(prior), float(priorU))

            print(toPrint)
            lossesT, losses_namesT = modelTePass(model, elbo, params, optimizer, logFile, testBatch=np.inf)
        
        classication_loss = classication_loss / params.alpha

    precision = 100*precision_k(y.data.cpu().numpy().squeeze(), preds.data.cpu().numpy().squeeze(), 5)
    if params.ss:
        return [precision[0], classication_loss, recon], ['Prec_1', 'BCELoss', 'lblLossTrain']
    else:
        return [precision[0], classication_loss], ['Prec_1', 'BCELoss']

  def modelTePass(model, elbo, params, optimizer, logFile, testBatch=5000):
    model.eval()
    classication_loss, total_loss, labelled_loss, unlabelled_loss, kl, recon, Lpred, Lgt, reconU  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    dataPts, XAll, ygt, ypred = 0, [], [], []
    reconFromY = 0.0

    for i, (x, y) in enumerate(params.validation):
        x, y = Variable(x).squeeze().float(), Variable(y).squeeze().float()
        dataPts +=x.shape[0]
        if dataPts > testBatch:
          break
        if params.cuda:
            x, y = x.cuda(device=0), y.cuda(device=0)

        U, _, reconAU, _, _ = elbo(x, temperature=params.temperature, normal=params.normal)
        L, klA, reconA, _ = elbo(x, y=y)
        lp, _, _, _= elbo(x, y=gumbel_multiSample(logits, params.temperature))
        logits, preds = model.classify(x)

        reconstruction = model.generate(y)
        diff = reconstruction - x

        # classication_loss = -torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()
        classication_loss = torch.nn.functional.binary_cross_entropy(preds, y)*y.shape[-1]
        loss = L + params.alpha * classication_loss + U

        total_loss += loss.data.cpu().numpy()
        labelled_loss += L.data.cpu().numpy()
        unlabelled_loss += U.data.cpu().numpy()
        classication_loss += classication_loss.data.cpu().numpy()#torch.mean((pred_idx.data == lab_idx.data).float())
        kl += klA
        recon += reconA
        reconU += reconAU
        ypred.append(preds.data.cpu().numpy().squeeze())
        ygt.append(y.data.cpu().numpy().squeeze())
        XAll.append(x.data.cpu().numpy().squeeze())
        lp, _, _, _= elbo(x, y=gumbel_multiSample(logits, params.temp))
        Lpred += lp.data.cpu().numpy()
        Lgt += L.data.cpu().numpy()
        reconFromY += torch.sum(torch.mul(diff, diff), dim=-1).data.cpu().numpy().mean()

    m = min(len(params.validation), i)
    ygt, ypred, XAll = np.concatenate(ygt, axis=0), np.concatenate(ypred, axis=0), np.concatenate(XAll, axis=0)
    P = 100*precision_k(ygt, ypred,5)
    if P[0] > params.bestP:
      params.bestP = P[0]
    # save_model(model, optimizer, params.epoch, params, "/model_best_test_" + params.mn + "_" + str(params.ss))
    # if classication_loss / m < params.best:
    #   params.best = classication_loss / m
    
    # toPrint = "[TEST]:Temp {:.3f}, Factor {:.3f}, Total Loss {:.2f}, Labelled Loss {:.2f}, KL {:.2f}, recon {:.2f}, unlabelled loss {:.2f}, classication_loss {:.2f}, best_p1 {}, best_bce {:.2f}".format(
    #       float(params.temp), params.reconFact.data.cpu().numpy(), float(total_loss / m), float(labelled_loss/ m), float(kl/m), float(recon/m), float(unlabelled_loss/ m), float(classication_loss/ m), params.bestP, params.best)
    toPrint = '[TEST] reconFromY {:.6} recon {:.6f}, reconU {:.6f} lblLossPred {:.2f}, lblLossGT {:.2f} '.format(\
    float(reconFromY/m), float(recon/m), float(reconU/m), Lpred / m, Lgt/m)
    toPrint += " || Prec Best " + str(params.bestP) + " Prec. " + str(P[0])+ " " + str(P[2]) + " " + str(P[4])

    logFile.write(toPrint + '\n')
    logFile.flush()
    print(toPrint)
    if params.ss:
          return [P[0], classication_loss / m, Lpred / m, Lgt/m], ['Prec_1_Test', 'BCELossTest', 'lblLossPred', 'lblLossGT']
    else:
          return [P[0], classication_loss / m], ['Prec_1_Test', 'BCELossTest']
