import pickle
import numpy as np
import pandas as pd
#from sklearn.utils import shuffle

with open('/mnt/home/lehieu1/IceCube/plot/iceprod/11900_hist.pkl','rb') as f:
    data = pickle.load(f)

print(data)
print(data.keys())
print(np.shape(data['true_zenith']))
