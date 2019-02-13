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

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

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
                        transform=flatten_bernoulli, target_transform=onehot(params.y_dim))
    mnist_valid = MNIST(location, train=False, download=True,
                        transform=flatten_bernoulli, target_transform=onehot(params.y_dim))

    def get_sampler(labels, n=None):
        # Only choose digits in y_dim
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(params.y_dim)]))

        # Ensure uniform distribution of labels
        np.random.shuffle(indices)
        indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in range(params.y_dim)])

        indices = torch.from_numpy(indices)
        sampler = SubsetRandomSampler(indices)
        return sampler

    # Dataloaders for MNIST
    
    labelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=2, pin_memory=params.cuda, sampler=get_sampler(mnist_train.train_labels.numpy(), labels_per_class))
    unlabelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=2, pin_memory=params.cuda, sampler=get_sampler(mnist_train.train_labels.numpy()))
    validation = torch.utils.data.DataLoader(mnist_valid, batch_size=batch_size, num_workers=2, pin_memory=params.cuda, sampler=get_sampler(mnist_valid.test_labels.numpy()))

    return labelled, unlabelled, validation

class Dataset(data.Dataset):
    def __init__(self, params, dtype, sp, scaler=None):
        self.sparse = sp

        if(self.sparse):
            self.x = sparse.load_npz('../datasets/' + params.data_set + '/x_' + dtype + '.npz').astype('float32')
            self.x = self.x/self.x.max()
            self.y = sparse.load_npz('../datasets/' + params.data_set + '/y_' + dtype + '.npz').astype('float32')
        else:
            if scaler is None:
                x_for_pp = np.load('../datasets/' + params.data_set + '/x_tr.npy')
                pp = MinMaxScaler()
                self.scaler = pp.fit(x_for_pp)
            else:
                self.scaler = scaler
            self.x = np.load('../datasets/' + params.data_set + '/x_' + dtype + '.npy').astype('float32')
            self.x = self.x/self.x.max()

	    self.y = np.load('../datasets/' + params.data_set + '/y_' + dtype + '.npy').astype('float32')
        self.maxX = self.x.max()
        self.x = (self.x - self.x.min())/(self.x.max() - self.x.min())
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
        # print("*** Getting Item ******")
        x = self.x[index, :]
        y = self.y[index, :]
        if self.sparse:
            x = x.todense()
            y = y.todense()
        X = torch.from_numpy(x.reshape((1, x.shape[-1])))#torch.load('data/' + ID + '.pt')
        y = torch.from_numpy(y.reshape((1, y.shape[-1])))#self.labels[ID]
        return X, y


def get_dataset(params):
    if params.data_set=="mnist":
        params.y_dim = 10
        params.x_dim = 784
        params.labelled, params.unlabelled, params.validation =  get_mnist(params)
    elif params.data_set=="amzn" or params.data_set=="rcv":
        print("TYPE 2")
        print("Loading dataset " + params.data_set)
        print("="*50)
        args = {'batch_size': params.mb,
            'shuffle': True,
            'num_workers': 0 }
        params.labelled = Dataset(params, "subs", 1)
        params.y_dim = params.labelled.getClasses()
        params.x_dim = params.labelled.getDims()
        params.labelled = data.DataLoader(params.labelled, **args)
        params.unlabelled = data.DataLoader(Dataset(params, "tr", 1), **args)
        params.maxX = Dataset(params, "tr", 1).maxX
        params.validation = data.DataLoader(Dataset(params, "te", 1), **args)

    else:# params.data_set=="delicious" or params.data_set == "bibtex":
        print("TYPE 3")
        print("Loading dataset " + params.data_set)
        print("="*50)
        params.labelled = Dataset(params, "new", 0) if params.new else Dataset(params, "subs", 0)
        scaler = params.labelled.getScaler()
        params.unlabelled =  Dataset(params, "new", 0) if  params.new else Dataset(params, "tr", 0, scaler)
        
        
        args = {'batch_size': params.mb,
            'shuffle': True,
            'num_workers': 2}
        argsL = {'batch_size': max(2, params.mb//(len(params.unlabelled)/len(params.labelled))),
            'shuffle': True,
            'num_workers': 1}


        params.y_dim = params.labelled.getClasses()
        params.x_dim = params.labelled.getDims()
        params.maxX = params.unlabelled.maxX

        params.unlabelled = data.DataLoader(params.unlabelled, **args)
        params.labelled = data.DataLoader(params.labelled, **args)
        params.validation = data.DataLoader(Dataset(params, "te", 0, scaler), **args)
    return params


