import numpy as np
from scipy import sparse
import time
try:
	x  = np.load('x_tr.npy')
	ty = np.load('y_tr.npy')
	sprse = 0
except:
	x  = sparse.load_npz('x_tr.npz')#.todense()
	ty = sparse.load_npz('y_tr.npz')#.todense()
	sprse = 1
ty_cp = ty.copy() 
ty = ty_cp.copy()
labels_num = np.sum(ty,axis=0)
removed_indices = []
print(ty.shape)
N = ty.shape[0]
nf = .05*N
n_now = N
labels_num = np.sum(ty,axis=0)
not_zero = np.argwhere(labels_num!=0)
def index_rows_by_exclusion_nptake(arr, i):
    """
    Return copy of arr excluding single row of position i using
    numpy.take function
    """
    return arr.take(range(i)+range(i+1,arr.shape[0]), axis=0)

# indices = np.random.randint(0, high=N, size=N)
trigger = 0
while (ty.shape[0] > nf):

    candidate = labels_num - ty[0]
    not_zero = np.argwhere(labels_num!=0)[0,:]
    #print(not_zero)
    if(len(np.argwhere(candidate[not_zero]==0))==0):
        print("="*10 + " "*5 + str(ty.shape[0]) + "/" + str(nf) +" "*5 + "="*10)
	labels_num = candidate
    	ty = ty[1:]
	x = x[1:]
	print('-'*10)
	trigger = 0
    else:
	trigger +=1
	if sprse:
		ty = sparse.vstack((ty[1:], ty[0]))
                x = sparse.vstack((x[1:], x[0]))
	else:
		ty = np.concatenate((ty[1:], ty[0].reshape((1, ty[0].shape[0]))), axis=0)
		x = np.concatenate((x[1:], x[0].reshape((1, x[0].shape[0]))), axis=0)
	print("Num labels offended: " +str(len(np.argwhere(candidate[not_zero]==0))))
    if trigger > ty.shape[0]/10:
	break
    #break
print(x.shape)
print(ty.shape)
if sprse:
	sparse.save_npz('x_subs', x)
	sparse.save_npz('y_subs', ty)
else:
	np.save('x_subs', x)
	np.save('y_subs', ty)
