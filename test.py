import pickle
import numpy as np
import pandas as pd
#from sklearn.utils import shuffle

with open('/mnt/scratch/lehieu1/training_files/iceprod_test/090721_000000_training.pkl','rb') as f:
    X,y,w,e,f,E,r = pickle.load(f)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

print(X)
print(y)

