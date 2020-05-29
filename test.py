import pickle
import numpy as np
from sklearn.utils import shuffle

nb_data = 150000
nb_dup = 10
nb_track_casc = nb_data/2
dup_times = int((nb_track_casc - nb_track_casc % nb_dup)/nb_dup)

with open('/mnt/scratch/lehieu1/training_files/with_energy/052920_000000_training.pkl','rb') as f:
    X,y,w,e,f,E = pickle.load(f)

print(E[0:10])

track_ind = np.where(y == 1)[0][:nb_dup]
casc_ind = np.where(y == 0)[0][:nb_dup]

def duplicate(array, track_ind, casc_ind, dup_times):
    return array[track_ind].tolist()*dup_times + array[casc_ind].tolist()*dup_times

X_data = duplicate(X, track_ind, casc_ind, dup_times)
y_data = duplicate(y, track_ind, casc_ind, dup_times)
w_data = duplicate(w, track_ind, casc_ind, dup_times)
e_data = duplicate(e, track_ind, casc_ind, dup_times)
f_data = duplicate(f, track_ind, casc_ind, dup_times)

w_data = np.ones(np.shape(w_data)).tolist()

X_data,y_data,w_data,e_data,f_data = shuffle(X_data,y_data,w_data,e_data,f_data)

#with open('/mnt/scratch/lehieu1/training_files/150000pts_10dup_weight1.pkl','wb') as f:
#    pickle.dump([X_data,y_data,w_data,e_data,f_data],f)
