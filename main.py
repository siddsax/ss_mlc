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
from models import DeepGenerativeModel, VariationalAutoencoder, StackedDeepGenerativeModel
from itertools import repeat, cycle
from torch.autograd import Variable
from inference import SVI, DeterministicWarmup, ImportanceWeightedSampler
from modelPass import modelTrPass, modelTePass
import argparse
from visualizer import Visualizer
from trainVAE import trainVAE
torch.manual_seed(1337)
np.random.seed(1337)


params = argparse.ArgumentParser(description='Process some integers.')
params.add_argument('--ss', dest='ss', type=int, default=1, help='1 to do semi-super, 0 for not doing it')
params.add_argument('--nrml', dest='normal', type=int, default=0, help='1 to do semi-super, 0 for not doing it')
params.add_argument('--ds', dest='data_set', type=str, default="mnist", help='mnist; delicious;')
params.add_argument('--mn', dest='name', type=str, default="", help='mnist; delicious;')
params.add_argument('--f', dest='feat', type=str, default="", help='mnist; delicious;')

params = params.parse_args()
params.cuda = torch.cuda.is_available()
print("CUDA: {}".format(params.cuda))

if __name__ == "__main__":
    viz = Visualizer(params)
    params.best = 1e10
    params.n_labels = 10
    # params.labelled, params.unlabelled, params.validation = get_mnist(params,location="./", batch_size=100, labels_per_class=100)
    params.bestP = 0.0
    params = get_dataset(params)
    params.alpha = 4.5#1 * len(params.unlabelled) / len(params.labelled)
    print(params.alpha)
    params.epochs = 2510
    params.step = 0
    if len(params.feat)==0:
       features = VariationalAutoencoder([params.xdim, 32, [256, 128]])#trainVAE(params, [params.xdim, 32, [256, 128]], epoch=100)
    else:
       if params.cuda:
	  features = torch.load(params.feat)
       else:
          features = torch.load(params.feat, map_location=lambda storage, loc: storage)
    model = StackedDeepGenerativeModel([params.xdim, params.n_labels, 32, [256, 128]], features)
    #model = DeepGenerativeModel([params.xdim, params.n_labels, 50, [600, 600]])
    if params.cuda:
        model = model.cuda()
    # , sampler=sampler) #,beta=beta)
    elbo = SVI(model, likelihood=binary_cross_entropy)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=1e-4, betas=(0.9, 0.999))

    # for epoch in range(params.epochs):
    #     params.epoch = epoch
    #     modelTrPass(model, optimizer, elbo, params)
    #     if epoch % 1 == 0:
    #         modelTePass(model, elbo, params)
    for epoch in range(params.epochs):
        params.epoch = epoch
        losses, losses_names = modelTrPass(model, optimizer, elbo, params)
        if epoch % 1 == 0:
            lossesT, losses_namesT = modelTePass(model, elbo, params, optimizer)
            losses += lossesT
            losses_names += losses_namesT

        lossDict = {}
        for key, val in zip(losses_names, losses):
            lossDict[key] = val
        viz.plot_current_losses(epoch, lossDict)
