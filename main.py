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
from modelPass import modelTrPass, modelTePass
import argparse
from visualizer import Visualizer
torch.manual_seed(1337)
np.random.seed(1337)


params = argparse.ArgumentParser(description='Process some integers.')
params.add_argument('--ss', dest='ss', type=int, default=1, help='1 to do semi-super, 0 for not doing it')
params.add_argument('--nrml', dest='normal', type=int, default=0, help='1 to do semi-super, 0 for not doing it')
params.add_argument('--ds', dest='data_set', type=str, default="mnist", help='mnist; delicious;')
params.add_argument('--mn', dest='name', type=str, default="", help='mnist; delicious;')
params.add_argument('--lm', dest='lm', type=int, default=0, help='mnist; delicious;')

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
    model = DeepGenerativeModel([params.xdim, params.n_labels, 50, [600, 600]])
    if params.cuda:
        model = model.cuda()
    # , sampler=sampler) #,beta=beta)
    elbo = SVI(model, likelihood=binary_cross_entropy)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    init = 0
    if(params.lm):
	print("================= Loading Model ============================")
	model, optimizer, init = load_model(model, 'saved_models/model_best_test_' + str(params.ss), optimizer)
    for epoch in range(init, params.epochs):
        params.epoch = epoch
        losses, losses_names = modelTrPass(model, optimizer, elbo, params)
        if epoch % 1 == 0:
            lossesT, losses_namesT = modelTePass(model, elbo, params, optimizer, testBatch=np.inf)
            losses += lossesT
            losses_names += losses_namesT

        lossDict = {}
        for key, val in zip(losses_names, losses):
            lossDict[key] = val
        viz.plot_current_losses(epoch, lossDict)

# 54.81946468 | 55.79
# | 55.82417846 Temp max(0.3, np.exp(-params.step*1e-4)) BCE
# | 56.01255894 Temp 1 BCE
# 54.22291756 | 56.23233914 Temp 1 BCE ST-gumbel-mc 
# 54.78807092 | 56.51491284 Temp 1 BCE ST-gumbel-mc
# 56.01255894 | 56.86028004 Temp 1 BCE ST-gumbel-mc Dropout
