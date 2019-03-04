import numpy as np
import scipy.io
import sys

path = sys.argv[1]

x_tr = np.load(path + '/x_tr.npy')
y_tr = np.load(path + '/y_tr.npy')
x_te = np.load(path + '/x_te.npy')
y_te = np.load(path + '/y_te.npy')
x_subs = np.load(path + '/x_subs.npy')
y_subs = np.load(path + '/y_subs.npy')

scipy.io.savemat(path + '/x_tr', mdict={'x_tr': x_tr})
scipy.io.savemat(path + '/y_tr', mdict={'y_tr': y_tr})
scipy.io.savemat(path + '/x_te', mdict={'x_te': x_te})
scipy.io.savemat(path + '/y_te', mdict={'y_te': y_te})
scipy.io.savemat(path + '/x_subs', mdict={'x_subs': x_subs})
scipy.io.savemat(path + '/y_subs', mdict={'y_subs': y_subs})


