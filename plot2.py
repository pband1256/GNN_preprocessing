import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import NullFormatter

def load_data(path, filename, label):
    data  = pd.read_csv(path+filename)
    if pd.api.types.is_string_dtype(data[label]):
        symbols = "['!\"#$%&()*/:;<=>?@[\]^_`{|}~\n]"
        array = np.array(data[label].str.replace(symbols, ""))

        # Convert to ndarray
        array = np.array([
            np.fromstring(j, dtype=np.float, sep=' ') for j in array
            ]).squeeze()
    else:
        array = np.array(data[label])
    return array

path  = "/mnt/home/lehieu1/IceCube/code/GNN/gnn_icecube/models/210707_11900_regr_logE_IC86/0/"
#path2 = "/mnt/home/lehieu1/IceCube/code/GNN/gnn_icecube/models/210621_11900_logE_regr/0/"
nb_files = 10

preds = load_data(path, "preds.csv", "prediction")
true  = load_data(path, "preds.csv", "truth")
#preds2, true2 = load_data(path2, "preds.csv")
tloss = load_data(path, "training_stats.csv", "train_tpr")
vloss = load_data(path, "training_stats.csv", "train_roc")
epoch = load_data(path, "training_stats.csv", "Epoch")/nb_files

plt.figure(figsize=(10,7))
plt.plot(epoch, tloss, label="Training")
plt.plot(epoch, vloss, label="Validation")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

savefolder = path
savename = "LossProgress"
plt.savefig("%s%s.png"%(savefolder,savename))

#plot_energy_slices(true, preds, use_fraction=True, bins=20, minenergy=hist_min, maxenergy=hist_max)
#plot_energy_slices(true2, preds2, use_fraction=True, bins=20, minenergy=hist_min, maxenergy=hist_max)



