import pickle
import numpy as np

with open('/mnt/scratch/lehieu1/training_files/052320_000000_training.pkl','rb') as f:
    X,y,w,e,f = pickle.load(f)

print(np.shape(X[0]))
print(np.shape(y))
print(np.shape(w))
print(np.shape(e))
print(np.shape(f))
