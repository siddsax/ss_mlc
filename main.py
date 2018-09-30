import torch
import numpy as np
import sys
from functools import reduce
from operator import __or__
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from datetime import datetime
# sys.path.append("../semi-supervised")
from utils import *
from models import DeepGenerativeModel
from itertools import repeat, cycle
from torch.autograd import Variable
from inference import SVI, DeterministicWarmup, ImportanceWeightedSampler
from modelPass import modelTrPass, modelTePass
import argparse
import os
from visualizer import Visualizer
torch.manual_seed(1337)
np.random.seed(1337)


params = argparse.ArgumentParser(description='Process some integers.')
params.add_argument('--ss', dest='ss', type=int, default=1, help='1 to do semi-super, 0 for not doing it')
params.add_argument('--oss', dest='oss', type=int, default=0, help='1 to ONLY do semi-super')
params.add_argument('--ld', dest='ld', type=int, default=0, help='1 to load model')
params.add_argument('--nrml', dest='normal', type=int, default=0, help='1 to do semi-super, 0 for not doing it')
params.add_argument('--ds', dest='data_set', type=str, default="mnist", help='mnist; delicious;')
params.add_argument('--zz', dest='name', type=str, default="", help='mnist; delicious;')
params.add_argument('--mn', dest='mn', type=str, default="", help='name')
params.add_argument('--lm', dest='lm', type=int, default=0, help='load model or not from the above name')
params.add_argument('--a', dest='alpha', type=float, default=5.5, help='mnist; delicious;')
params.add_argument('--mb', dest='mb', type=int, default=100, help='mnist; delicious;')
params.add_argument('--f', dest='factor', type=float, default=5, help='mnist; delicious;')
params.add_argument('--t', dest='type', type=float, default=5, help='mnist; delicious;')

params = params.parse_args()
params.cuda = torch.cuda.is_available()
print("CUDA: {}".format(params.cuda))

if __name__ == "__main__":
    lr = 1e-3
    viz = Visualizer(params)
    if not os.path.exists('logs'):
    	os.makedirs('logs')
    logFile = params.mn if len(params.mn) else str(datetime.now())
    print("=================== Name of logFile is =======    " + logFile + "     ==========")
    logFile = open('logs/' + logFile + '.logs', 'w+')
    dgm = open('models/dgm.py').read()
    logFile.write(" WE are running on " + str(params.ss) + "    ====\n")
    logFile.write(" WE are having LR " + str(lr) + "    ====\n")    
    logFile.write('=============== DGM File ===================\n\n')
    logFile.write(dgm)
    logFile.write('\n\n=============== VAE File ===================\n\n')
    logFile.write(open('models/vae.py').read())
    
    params.temp = 1.0
    params.reconFact = 1.0
    params.n_labels = 10
    # params.labelled, params.unlabelled, params.validation = get_mnist(params,location="./", batch_size=100, labels_per_class=100)
    params.bestP = 0.0
    params.bestR = 1e10
    params = get_dataset(params)
    print(params.alpha)
    params.epochs = 2510
    params.step = 0
    model = DeepGenerativeModel([params.xdim, params.n_labels, 50, [400, 300]], params)
    if params.cuda:
        model = model.cuda()
    # , sampler=sampler) #,beta=beta)
    elbo = SVI(model, params, likelihood=binary_cross_entropy)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.999))
    init = 0

    if(params.lm==1):
        print("================= Loading Model 1 ============================")
        model, optimizer, init = load_model(model, 'saved_models/model_best_class_' + params.mn + "_" + str(params.ss), optimizer)
    elif(params.lm==2):
        print("================= Loading Model 2 ============================")
        model, optimizer, init = load_model(model, 'saved_models/model_best_regen_' + params.mn + "_" + str(params.ss), optimizer)

    for epoch in range(init, params.epochs):
        params.epoch = epoch
        losses, losses_names = modelTrPass(model, optimizer, elbo, params, logFile, viz=viz)
        # if epoch % 1 == 0:
        #     lossesT, losses_namesT = modelTePass(model, elbo, params, optimizer, logFile, testBatch=np.inf)
        #     losses += lossesT
        #     losses_names += losses_namesT

        # lossDict = {}
        # for key, val in zip(losses_names, losses):
        #     lossDict[key] = val
        # viz.plot_current_losses(epoch, lossDict)
        print("="*100)
