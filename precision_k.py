import numpy as np
import scipy.io as sio
from sklearn.metrics import f1_score

def f1_measure(true_mat, preds, average='binary'):
    f1_scores = f1_score(true_mat, preds, average=average)
    return f1_scores

def precision_k(true_mat, score_mat,k):
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)

    score_mat = np.copy(backup)
    for i in range(rank_mat.shape[0]):
        score_mat[i][rank_mat[i, :-(k+1)]] = 0
    score_mat = np.ceil(score_mat)
    mat = np.multiply(score_mat, true_mat)
    num = np.sum(mat,axis=1)
    p = np.mean(num/(k+1))

    # out = ""
    # for i in range(len(p)):
    #     out += str(i) + ":" + str(p[i]) + " " 
    # print(out)
    return 100*np.around(p, decimals=4)

