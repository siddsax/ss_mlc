import numpy as np
import scipy.io
x_tr = np.load('x_tr.npy')
y_tr = np.load('y_tr.npy')
x_te = np.load('x_te.npy')
y_te = np.load('y_te.npy')
x_subs = np.load('x_subs.npy')
y_subs = np.load('y_subs.npy')

scipy.io.savemat('x_tr', mdict={'x_tr': x_tr})
scipy.io.savemat('y_tr', mdict={'y_tr': y_tr})
scipy.io.savemat('x_te', mdict={'x_te': x_te})
scipy.io.savemat('y_te', mdict={'y_te': y_te})
scipy.io.savemat('x_subs', mdict={'x_subs': x_subs})
scipy.io.savemat('y_subs', mdict={'y_subs': y_subs})

