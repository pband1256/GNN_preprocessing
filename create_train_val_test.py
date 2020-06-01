import pickle
import glob
import argparse
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--nb_train",type=int,default=10,
                    dest="nb_train",help="number of training files")
parser.add_argument("-v", "--nb_val",type=int,default=10,
                    dest="nb_val",help="number of validation files")
parser.add_argument("-e", "--nb_test",type=int,default=10,
                    dest="nb_test",help="number of test files")
parser.add_argument("-i", "--input",type=str,default="/mnt/scratch/lehieu1/training_files",
                    dest="inp",help="path to directory of input pickle files")
parser.add_argument("-o", "--output",type=str,default="/mnt/scratch/lehieu1/training_files/processed",
                    dest="out",help="path to directory of output pickle files")
args = parser.parse_args()

# Masking non-coordinate features of feature array
def mask_features(data,feature_ind=(3,4,5)):
    for i in range(np.shape(data)[0]):
        data[i][:,feature_ind] = 0
    return data

# Masking coordinates of feature array (padding inactive DOMs with 0 for pulse data)
def mask_coordinates(data):
    coords = []
    batch_size = np.shape(data)[0]
    for i in range(batch_size):
        coords.append(data[i][:,0:3])
    coords_list = np.unique(np.concatenate(coords),axis=0)
    
    for i in range(batch_size):
        inactive_DOMs = np.array([x for x in set(tuple(x) for x in coords_list) ^ set(tuple(x) for x in coords[i])])
        data[i] = np.concatenate([np.concatenate([inactive_DOMs,np.zeros(np.shape(inactive_DOMs))],axis=1),data[i]])
    return data

# Open pickled .i3 files one by one and concatenate all data into master arrays
def pickleList(fileList):
    first = True
    for fileName in fileList:
        try:
            with open(fileName,'rb') as f:
                X, y, weights, event_id, filenames, energy = pickle.load(f)
            if first == True:
                X_all = X
                y_all = y
                w_all = weights
                e_all = event_id
                f_all = filenames
                E_all = energy
                first = False
            else:
                X_all = np.concatenate((X_all,X))
                y_all = np.concatenate((y_all,y))
                w_all = np.concatenate((w_all,weights))
                e_all = np.concatenate((e_all,event_id))
                f_all = np.concatenate((f_all,filenames))
                E_all = np.concatenate((E_all,energy))
        except ValueError or EOFError as e:
            print("Error: file "+fileName+" failed to pickle correctly. Skipping file")
            print(e)
            continue

    ####### Setting weights to 1
    w_all = np.ones(np.shape(w_all))
    ####### Masking features
    #X_all = mask_features(X_all)
    ####### Masking coordinates
    X_all = mask_coordinates(X_all)

    return [X_all,y_all,w_all,e_all,f_all,E_all]

# Shuffling ALL files in folder to make sure there's no systematic problem. Probably overkill.
nb_total = args.nb_train + args.nb_val + args.nb_test
total_file = glob.glob(args.inp + '/*.pkl')
random.shuffle(total_file)

train_file = total_file[0:args.nb_train]
val_file = total_file[args.nb_train:nb_total-args.nb_test]
test_file = total_file[nb_total-args.nb_test:nb_total]

with open(args.out + '/train_file.pkl',"wb") as f:
    pickle.dump(pickleList(train_file),f)

with open(args.out + '/val_file.pkl',"wb") as f:
    pickle.dump(pickleList(val_file),f)

with open(args.out + '/test_file.pkl',"wb") as f:
    pickle.dump(pickleList(test_file),f)
