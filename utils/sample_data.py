import numpy as np
from scipy import sparse
import time
import random
import argparse

params = argparse.ArgumentParser(description='Process some integers.')
params.add_argument('--p', dest='percentage', required=True, type=float, help='percentage to sample data')
params.add_argument('--d', dest='dataset', required=True, type=str, help='directory of dataset')
params = params.parse_args()

try:
	x  = np.load(params.dataset + '/x_tr.npy')
	ty = np.load(params.dataset + '/y_tr.npy')
	sprse = 0
except:
	x  = sparse.load_npz(params.dataset + '/x_tr.npz')
	ty = sparse.load_npz(params.dataset + '/y_tr.npz')
	sprse = 1

# Shuffling
index_shuf = range(x.shape[0])
random.shuffle(index_shuf)

x = x[index_shuf]
ty = ty[index_shuf]

N = ty.shape[0]
nf = int(params.percentage*N/100.0)
x = x[:nf]
ty = ty[:nf]

print(x.shape, ty.shape)

if sprse:
	sparse.save_npz(params.dataset + 'x_subs', x)
	sparse.save_npz(params.dataset + 'y_subs', ty)
else:
	np.save(params.dataset + '/x_subs', x)
	np.save(params.dataset + '/y_subs', ty)
