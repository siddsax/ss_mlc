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
torch.manual_seed(1337)
np.random.seed(1337)


params = argparse.ArgumentParser(description='Process some integers.')
params.add_argument('--ss', dest='ss', type=int, default=1, help='1 to do semi-super, 0 for not doing it')
params.add_argument('--nrml', dest='normal', type=int, default=0, help='1 to do semi-super, 0 for not doing it')
params = params.parse_args()
params.cuda = torch.cuda.is_available()
print("CUDA: {}".format(params.cuda))

if __name__ == "__main__":
    params.best = 0.0
    params.n_labels = 10
    params.labelled, params.unlabelled, params.validation = get_mnist(params,location="./", batch_size=100, labels_per_class=100)
    params.alpha = 0.1 * len(params.unlabelled) / len(params.labelled)
    params.epochs = 251
    params.step = 0
    model = DeepGenerativeModel([784, 10, 50, [600, 600]])
    if params.cuda:
        model = model.cuda()
    # , sampler=sampler) #,beta=beta)
    elbo = SVI(model, likelihood=binary_cross_entropy)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=3e-4, betas=(0.9, 0.999))

    for epoch in range(params.epochs):
        params.epoch = epoch
        modelTrPass(model, optimizer, elbo, params)
        if epoch % 1 == 0:
            modelTePass(model, elbo, params)