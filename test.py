import pickle
import numpy as np
import pandas as pd
#from sklearn.utils import shuffle

with open('/mnt/scratch/lehieu1/training_files/11900_regr_logE/processed/test_file.pkl','rb') as f:
    X,y,w,e,f,E = pickle.load(f)

print(pd.DataFrame([e,E]).transpose())
e = np.array([e])
hist, bins = np.histogram(e,bins=np.arange(np.min(e),np.max(e)))
print(hist)


