import pickle
import glob
import argparse
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--nb_train",type=int,default=10,
                    dest="nb_train",help="number of training sample")
parser.add_argument("-v", "--nb_val",type=int,default=10,
                    dest="nb_val",help="number of validation sample")
parser.add_argument("-e", "--nb_test",type=int,default=10,
                    dest="nb_test",help="number of test sample")
parser.add_argument("-i", "--input",type=str,default="/mnt/scratch/lehieu1/training_files",
                    dest="inp",help="path to directory of input pickle files")
parser.add_argument("-o", "--output",type=str,default="/mnt/scratch/lehieu1/training_files/processed",
                    dest="out",help="path to directory of output pickle files")
args = parser.parse_args()

def pickleList(fileList):
    first = True
    for fileName in fileList:
        try:
            with open(fileName,'rb') as f:
                X, y, weights, event_id, filenames = pickle.load(f)
            if first == True:
                X_all = X
                y_all = y
                w_all = weights
                e_all = event_id
                f_all = filenames
                first = False
            else:
                X_all = np.concatenate((X_all,X))
                y_all = np.concatenate((y_all,y))
                w_all = np.concatenate((w_all,weights))
                e_all = np.concatenate((e_all,event_id))
                f_all = np.concatenate((f_all,filenames))
        except ValueError or EOFError as e:
            print("Error: file "+fileName+" failed to pickle correctly. Skipping file")
            print(e)
            continue
    return [X_all,y_all,w_all,e_all,f_all]

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
