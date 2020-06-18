# some_file.py
import sys
# sys.path.insert(1, '/home/users/hieule/code/GNN/gnn_icecube/src')
sys.path.insert(1, '/home/joyvan/GNN/gnn_icecube/src')
import model

import pickle
import numpy as np
from sklearn.utils import shuffle

# with open('/mnt/scratch/lehieu1/training_files/processed/equal_labels_nocuts/train_file.pkl','rb') as f:
# with open('/data/icecube/hieule/training_files/processed/nocuts_multi/train_file_10.pkl','rb') as f:
with open('/home/jovyan/val_file.pkl','rb') as f:
    X,y,w,e,f,E = pickle.load(f)

w = np.ones(np.shape(w)).tolist()

X_data,y_data,w_data,e_data,f_data = shuffle(X_data,y_data,w_data,e_data,f_data)

#with open('/mnt/scratch/lehieu1/training_files/150000pts_10dup_weight1.pkl','wb') as f:
#    pickle.dump([X_data,y_data,w_data,e_data,f_data],f)
