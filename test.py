import pickle
import numpy as np
from sklearn.utils import shuffle

with open('/mnt/scratch/lehieu1/training_files/processed/nocuts_multi_5050/train_file_10.pkl','rb') as f:
    X,y,w,e,f,E = pickle.load(f)

print(np.size(E))

#X_data,y_data,w_data,e_data,f_data = shuffle(X_data,y_data,w_data,e_data,f_data)

#with open('/mnt/scratch/lehieu1/training_files/150000pts_10dup_weight1.pkl','wb') as f:
#    pickle.dump([X_data,y_data,w_data,e_data,f_data],f)
