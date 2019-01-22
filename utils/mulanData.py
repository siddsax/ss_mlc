import os
import sys
import numpy as np
from skmultilearn.dataset import load_from_arff

path_to_arff_file = sys.argv[1]
label_count = int(sys.argv[2])
label_location="end"
arff_file_is_sparse = False

X, y = load_from_arff(
    path_to_arff_file,
    label_count=label_count,
    label_location=label_location,
    load_sparse=arff_file_is_sparse
)

dirN = '/'.join(path_to_arff_file.split('/')[:-1])

X, y = X.todense(), y.todense()

if "train" in path_to_arff_file:
    np.save(dirN + '/x_tr', X)
    np.save(dirN + '/y_tr', y)
else:
    np.save(dirN + '/x_te', X)
    np.save(dirN + '/y_te', y)
    

