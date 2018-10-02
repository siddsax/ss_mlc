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
import cPickle
from visualizer import Visualizer


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
params.add_argument('--cY', dest='cY', type=str, default="", help='custom labels')
params.add_argument('--numC', dest='numC', type=int, default=30, help='custom labels')

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
    params.bestR = 1e10
    params.bestP = 0.0
    params.epoch = 0
    params.step = 0
    params = get_dataset(params)
    params.alpha = params.alpha#1 * len(params.unlabelled) / len(params.labelled)
    model = DeepGenerativeModel([params.xdim, params.n_labels, 100, [600, 200]], params)
    if params.cuda:
        model = model.cuda()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.999))
    elbo = SVI(model, params, likelihood=binary_cross_entropy)

    if(params.lm==1):
        print("================= Loading Model 1 ============================")
        model, optimizer, init = load_model(model, 'saved_models/model_best_class_' + params.mn + "_" + str(params.ss), optimizer)
    elif(params.lm==2):
        print("================= Loading Model 2 ============================")
        model, optimizer, init = load_model(model, 'saved_models/model_best_regen_' + params.mn + "_" + str(params.ss), optimizer)
    modelTePass(model, elbo, params, optimizer, logFile, testBatch=5000)

    y_tr = np.load('datasets/'+ params.data_set + '/y_tr.npy')
# - ------ legacy sampling ----------------------------------------------
    if len(params.cY)==0:
        if False:
            new_y = np.zeros(np.shape(y_tr))
            label_counts = np.sum(y_tr, axis=0)
            lives = label_counts.max() - label_counts
            x = np.sum(y_tr, axis=1)
            for i in range(np.shape(y_tr)[0]):
                labels = np.argwhere(lives>0)[:,0].astype(int)
                print("{}/{}".format(i, np.shape(y_tr)[0]))
                
                fin_labels = np.random.choice(labels, int(x[i]), replace=False, p=lives[labels]/lives.sum())
                new_y[i, fin_labels] = 1
                label_countsNew = np.sum(new_y, axis=0) + label_counts
                lives = label_countsNew.max() - label_countsNew
        else:
            new_y = np.zeros(np.shape(y_tr))
            label_counts = np.sum(y_tr, axis=0)
            lives = 10*(label_counts - label_counts.max()/15.77)
            # print(lives)
            # exit()
            x = np.sum(y_tr, axis=1).tolist()
            with open('datasets/' + params.data_set + '/labelling/classifier_' + str(params.numC)  + '.pkl', 'rb') as fid:
                kmeans = cPickle.load(fid)
            adjacency_mat = np.load('datasets/' + params.data_set + '/labelling/adjacency_mat.npy')
            clusters = kmeans.predict(adjacency_mat)
            num_clusters = int(np.max(clusters))
            lives[np.argwhere(lives>0)] = 0
            clusters[np.argwhere(lives==0)] = num_clusters + 1 
            data_pts_num = []
            data_pts = []
            for i in range(int(num_clusters)):
                data_pts.append(np.argwhere(clusters==i))           
                data_pts_num.append(len(data_pts[i]))


            data = 0
            priority_list = []
            stuck_count = 0
            while(np.sum(lives) < 0 and data < y_tr.shape[0]):
                if(len(priority_list)):
                    clst_num = priority_list[0]
                    priority_list.remove(clst_num)
                else:    
                    clst_num = np.random.randint(0, high=num_clusters)
                num_labels = np.random.choice(x)
                if(num_labels>data_pts_num[clst_num]):
                    if(stuck_count>10):
                        stuck_count = 0
                    else:
                        stuck_count+=1
                        priority_list.append(clst_num)
                        print(" ---- stuck ---- at {1} for {0} ----".format(num_labels, clst_num))
                        continue
                else:
                    x = np.delete(x, np.argwhere(x==num_labels)[0])
                    fin_labels = np.random.choice(data_pts[clst_num][:,0], int(num_labels), replace=False)
                    lives[fin_labels] += 1
                    clusters[np.argwhere(lives==0)] = num_clusters + 1
                    data_pts_num = []
                    data_pts = []
                    for i in range(num_clusters):
                        data_pts.append(np.argwhere(clusters==i))           
                        data_pts_num.append(len(data_pts[i]))

                    new_y[data, fin_labels] = 1
                    data+=1
                    print(data)

        params.cY = new_y#np.random.rand(100,params.n_labels)
        np.save('datasets/'+ params.data_set + '/y_only_new.npy', params.cY)
    else:
        params.cY = np.load(params.cY)
    customY = torch.autograd.Variable(torch.from_numpy(params.cY))
    newY = np.concatenate((y_tr, params.cY), axis=0)
    np.save('datasets/'+ params.data_set + '/y_new.npy', newY)
    if torch.cuda.is_available():
        customY = customY.cuda()
    reconstruction = model.generate(customY)*float(params.maxX)
    reconstruction = reconstruction.data.cpu().numpy()
   
    
    Xtr = np.load('datasets/'+ params.data_set + '/x_tr.npy')
    newX = np.concatenate((Xtr, reconstruction), axis=0)
    randX = np.concatenate((Xtr, np.random.rand(reconstruction.shape[0], reconstruction.shape[1])), axis=0)
    
    np.save('datasets/'+ params.data_set + '/x_new.npy', newX)
    np.save('datasets/'+ params.data_set + '/x_rand.npy', randX)

# ---------------------------------------------------------------------

# ---------- Cluster Sampling ---------------------------------------
# ---------------------------------------------------------------------
