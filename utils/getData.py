import torch
import numpy as np
import sys
from functools import reduce
from operator import __or__
from sklearn.preprocessing import MinMaxScaler
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from utils import *
from scipy import sparse
def get_mnist(params, location="./", batch_size=64, labels_per_class=100):


    flatten_bernoulli = lambda x: transforms.ToTensor()(x).view(-1).bernoulli()
    mnist_train = MNIST(location, train=True, download=True,
                        transform=flatten_bernoulli, target_transform=onehot(params.n_labels))
    mnist_valid = MNIST(location, train=False, download=True,
                        transform=flatten_bernoulli, target_transform=onehot(params.n_labels))

    def get_sampler(labels, n=None):
        # Only choose digits in n_labels
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(params.n_labels)]))

        # Ensure uniform distribution of labels
        np.random.shuffle(indices)
        indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in range(params.n_labels)])

        indices = torch.from_numpy(indices)
        sampler = SubsetRandomSampler(indices)
        return sampler

    # Dataloaders for MNIST
    labelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=2, pin_memory=params.cuda,
                                           sampler=get_sampler(mnist_train.train_labels.numpy(), labels_per_class))
    unlabelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=2, pin_memory=params.cuda,
                                             sampler=get_sampler(mnist_train.train_labels.numpy()))
    validation = torch.utils.data.DataLoader(mnist_valid, batch_size=batch_size, num_workers=2, pin_memory=params.cuda,
                                             sampler=get_sampler(mnist_valid.test_labels.numpy()))

    return labelled, unlabelled, validation

class Dataset(data.Dataset):
    def __init__(self, params, dtype, scaler=None):
        if scaler is None:
            x_for_pp = np.load('datasets/' + params.data_set + '/x_tr.npy')
            pp = MinMaxScaler()
            self.scaler = pp.fit(x_for_pp)
        else:
            self.scaler = scaler
        self.x = self.scaler.transform(np.load('datasets/' + params.data_set + '/x_' + dtype + '.npy')).astype('float32')
        self.y = np.load('datasets/' + params.data_set + '/y_' + dtype + '.npy').astype('float32')

    def __len__(self):
        return self.x.shape[0]

    def getClasses(self):
        return self.y.shape[-1]
    def getDims(self):
        return self.x.shape[-1]
    def getScaler(self):
        return self.scaler

    def __getitem__(self, index):
        # Select sample

        # Load data and get label
        X = torch.from_numpy(self.x[index, :].reshape((1, self.x.shape[-1])))#torch.load('data/' + ID + '.pt')
        y = torch.from_numpy(self.y[index, :].reshape((1, self.y.shape[-1])))#self.labels[ID]
        # import pdb
        # pdb.set_trace()

        return X, y

def get_dataset(params):
    if params.data_set=="mnist":
        params.labelled, params.unlabelled, params.validation =  get_mnist(params)
        params.n_labels = 10
        params.xdim = 784
    else:
        params.labelled = Dataset(params, "subs")
        scaler = params.labelled.getScaler()
        params.unlabelled, params.validation = Dataset(params, "tr", scaler), Dataset(params, "te", scaler)
        params.n_labels = params.labelled.getClasses()
        params.xdim = params.labelled.getDims()
    return params

