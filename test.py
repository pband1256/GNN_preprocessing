import pickle
import numpy as np
#from sklearn.utils import shuffle

with open('/mnt/scratch/lehieu1/training_files/processed/diffuse_final/test_file.pkl','rb') as f:
    X,y,w,e,f,E = pickle.load(f)

print(np.size(y)-np.sum(y))
print(np.shape(X))
print(np.shape(X[0]))
print(np.shape(E))

#X_data,y_data,w_data,e_data,f_data = shuffle(X_data,y_data,w_data,e_data,f_data)

#with open('/mnt/scratch/lehieu1/training_files/150000pts_10dup_weight1.pkl','wb') as f:
#    pickle.dump([X_data,y_data,w_data,e_data,f_data],f)
