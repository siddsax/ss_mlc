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

class CombineDataset(data.Dataset):
    def __init__(self, dataLrg, dataSml):
        self.data1 = dataLrg#call first instance
        self.data2 = dataSml#call second instance
        self.size1 = len(dataLrg)
        self.size2 = len(dataSml)

    def __len__(self):
        return self.size1

    def __getitem__(self,index):
        return self.data1[index], self.data2[index%self.size2]

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
    def __init__(self, params, dtype, sp, scaler=None):
        self.sparse = sp

        if(self.sparse):
            self.x = sparse.load_npz('datasets/' + params.data_set + '/x_' + dtype + '.npz').astype('float32')
            self.x = self.x/self.x.max()
            self.y = sparse.load_npz('datasets/' + params.data_set + '/y_' + dtype + '.npz').astype('float32')
        else:
            if scaler is None:
                x_for_pp = np.load('datasets/' + params.data_set + '/x_tr.npy')
                pp = MinMaxScaler()
                self.scaler = pp.fit(x_for_pp)
            else:
                self.scaler = scaler
            self.x = np.load('datasets/' + params.data_set + '/x_' + dtype + '.npy').astype('float32')
            self.x = self.x/self.x.max()

	    self.y = np.load('datasets/' + params.data_set + '/y_' + dtype + '.npy').astype('float32')
        self.maxX = self.x.max()
        print(self.x.shape, self.y.shape)
    
    def __len__(self):
        return self.x.shape[0]

    def getClasses(self):
        return self.y.shape[-1]
    def getDims(self):
        return self.x.shape[-1]
    def getScaler(self):
        if self.sparse:
            print("** cant scale sparse data **")
            exit()
        else:
            return self.scaler

    def __getitem__(self, index):
        x = self.x[index, :]
        y = self.y[index, :]
        if self.sp:
            x = x.todense()
            y = y.todense()
        X = torch.from_numpy(x.reshape((1, x.shape[-1])))
        y = torch.from_numpy(y.reshape((1, y.shape[-1])))
        return X, y


def get_dataset(params):

    print("Loading dataset " + params.data_set)
    print("="*50)
    args = {'batch_size': params.mb, 'shuffle': True, 'num_workers': 2}
    params.dataTr = Dataset(params, "subs", 0)
    params.n_labels = params.dataTr.getClasses()
    params.xdim = params.dataTr.getDims()
    params.dataTr = data.DataLoader(params.dataTr, **args)
    params.dataTe = data.DataLoader(Dataset(params, "te", 0), **args)
    return params

