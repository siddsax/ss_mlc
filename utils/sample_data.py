import numpy as np
from scipy import sparse
import time
import random

try:
	x  = np.load('x_tr.npy')
	ty = np.load('y_tr.npy')
	sprse = 0
except:
	x  = sparse.load_npz('x_tr.npz')
	ty = sparse.load_npz('y_tr.npz')
	sprse = 1

index_shuf = random.shuffle(range(x.shape[0]))
x = x[index_shuf]
ty = ty[index_shuf]

labels_num = np.sum(ty,axis=0)
removed_indices = []
N = ty.shape[0]
nf = .05*N
n_now = N
labels_num = np.sum(ty,axis=0)
not_zero = np.argwhere(labels_num!=0)

print(ty.shape)

trigger = 0
while (ty.shape[0] > nf):

    candidate = labels_num - ty[0]
    not_zero = np.argwhere(labels_num!=0)[0,:]

    # if(len(np.argwhere(candidate[not_zero]==0))==0):
	print("="*10 + " "*5 + str(ty.shape[0]) + "/" + str(nf) +" "*5 + "="*10)
	labels_num = candidate
	ty = ty[1:]
	x = x[1:]
	print('-'*10)
	trigger = 0
	# else:
	# 	trigger +=1	
	# 	if sprse:
	# 		ty = sparse.vstack((ty[1:], ty[0]))
	# 		x = sparse.vstack((x[1:], x[0]))
	# 	else:
	# 		ty = np.concatenate((ty[1:], ty[0].reshape((1, ty[0].shape[0]))), axis=0)
	# 		x = np.concatenate((x[1:], x[0].reshape((1, x[0].shape[0]))), axis=0)
	# 	print("Num labels offended: " +str(len(np.argwhere(candidate[not_zero]==0))))
    
	if trigger > ty.shape[0]/10:
		break

print(x.shape, ty.shape)

if sprse:
	sparse.save_npz('x_subs', x)
	sparse.save_npz('y_subs', ty)
else:
	np.save('x_subs', x)
	np.save('y_subs', ty)
