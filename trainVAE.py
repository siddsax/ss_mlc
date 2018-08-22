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
from layers import *
torch.manual_seed(1337)
np.random.seed(1337)


params = argparse.ArgumentParser(description='Process some integers.')
params.add_argument('--ss', dest='ss', type=int, default=1, help='1 to do semi-super, 0 for not doing it')
params.add_argument('--nrml', dest='normal', type=int, default=0, help='1 to do semi-super, 0 for not doing it')
params.add_argument('--ds', dest='data_set', type=str, default="mnist", help='mnist; delicious;')
params.add_argument('--mn', dest='name', type=str, default="", help='mnist; delicious;')
params.add_argument('--f', dest='feat', type=str, default="", help='mnist; delicious;')
params.add_argument('--ep', dest='ep', type=int, default=100, help='mnist; delicious;')

params = params.parse_args()
params.cuda = torch.cuda.is_available()
print("CUDA: {}".format(params.cuda))

def trainVAE(params, dims, epochs=100):
	[x_dim, z_dim, h_dim] = dims
	model = VariationalAutoencoder(dims)
	if params.cuda:
        	model = model.cuda()
	gaussian = GaussianSample(10, 1)
	z, mu, log_var = gaussian(Variable(torch.ones(1, 10)))
	optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
	train, validation = params.unlabelled, params.validation
	for epoch in range(epochs):
    		model.train()
    		total_loss = 0
    		for (u, _) in train:
        		u = Variable(u)
			u = Variable(u.cuda(device=0)) if params.cuda else Variable(u)

        		reconstruction = model(u)
        		likelihood = -binary_cross_entropy(reconstruction, u)
        		elbo = likelihood - model.kl_divergence
        
        		L = -torch.mean(elbo)

        		L.backward()
        		optimizer.step()
        		optimizer.zero_grad()

        		total_loss += L.data[0]

    		m = len(train)

   	 	if epoch % 10 == 0:
        		print("Epoch: {}\tL: {}".format(epoch, total_loss/m))
	return model

if __name__ == "__main__":
        params.n_labels=10
        params = get_dataset(params)
        model = trainVAE(params, [params.xdim, 32, [256, 128]], epochs=params.ep)
        torch.save(model, "vae_saved.pt")
