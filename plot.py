import numpy as np
import pandas as pd
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import NullFormatter

path  = "/mnt/home/lehieu1/IceCube/code/GNN/gnn_icecube/models/210707_11900_regr_logE_IC86/0/"
path2 = "/mnt/home/lehieu1/IceCube/code/GNN/gnn_icecube/models/210621_11900_logE_regr/0/"

def load_data(path):
    data  = pd.read_csv(path+"preds.csv")
    symbols = "['!\"#$%&()*/:;<=>?@[\]^_`{|}~\n]"
    preds = np.array(data['prediction'].str.replace(symbols, ""))
    true  = np.array(data['truth'].str.replace(symbols, ""))

    # Convert to ndarray
    preds = np.array([
        np.fromstring(i, dtype=np.float, sep=' ') for i in preds
        ]).squeeze()
    true  = np.array([
        np.fromstring(j, dtype=np.float, sep=' ') for j in true
        ]).squeeze()
    return preds, true

preds, true  = load_data(path)
preds2, true2 = load_data(path2)

path3 = "/mnt/scratch/lehieu1/training_files/11900_SplineMPE/processed/train_file_1.pkl"
with open(path3, "rb") as f:
    _,_,_,_,_,true3,preds3 = pickle.load(f)
true3 = np.log10(true3)
preds3 = np.log10(preds3)
bias = np.mean(preds3-true3)
preds3 -= bias
#print(pd.DataFrame(true3).describe())
#print(pd.DataFrame(preds3).describe())

# Plotting
plt.clf()
nullfmt = NullFormatter()

# definitions for the axes
left, width = 0.1, 0.6
bottom, height = 0.1, 0.6
bottom_h = left_h = left + width + 0.02
rect_hist  = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]
# start with a rectangular Figure
plt.figure(figsize=(8, 7))
axHist  = plt.axes(rect_hist)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)
# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

# the scatter plot:
hist_min = np.min(true)
hist_max = np.max(true)
bins = np.logspace(np.log10(hist_min), np.log10(hist_max), 100)
im = axHist.hist2d(preds, true, norm=colors.LogNorm(), bins=(bins,bins))
#bins = np.linspace(hist_min, hist_max, 100)
#im = axHist.hist2d(preds, true, bins=(bins,bins))
axHist.plot(bins, bins, color='r')

# now determine nice limits by hand:
axHistx.hist(preds, bins=bins)
axHisty.hist(true,  bins=bins, orientation='horizontal')
axHistx.set_xlim(axHist.get_xlim())
axHisty.set_ylim(axHist.get_ylim())

# Style
axHist.set_xlabel("Prediction")
axHist.set_ylabel("Truth")
#axHist.set_xscale('log')
#axHist.set_yscale('log')
#axHistx.set_xscale('log')
#axHistx.set_yscale('log')
#axHisty.set_xscale('log')
#axHisty.set_yscale('log')
axHistx.tick_params(labelbottom=False)
axHisty.tick_params(labelleft=False)
plt.suptitle("Prediction histogram (log E)")
#cax = plt.axes([0.27, 0.8, 0.5, 0.05])
#plt.colorbar(im, cax=cax)

#Save
plotfile = path+'test_hist_E.png'
plt.savefig(plotfile)
plt.clf()


###########################
######               ######
###### Jessie's code ######
######               ######
###########################

def plot_energy_slices(truth, nn_reco, \
                       use_fraction = False, use_old_reco = False, old_reco=None,\
                       bins=10,label="NN reco", minenergy=0.,maxenergy=60.,\
                       save=False,savefolder=None):
    """Plots different energy slices vs each other (systematic set arrays)
    Receives:
        truth= array with truth labels
                (contents = [energy], shape = number of events)
        nn_reco = array that has NN predicted reco results
                    (contents = [energy], shape = number of events)
        use_fraction = bool, use fractional resolution instead of absolute, where (reco - truth)/truth
        use_old_reco = bool, True if you want to compare to another reconstruction (like pegleg)
        old_reco = optional, array of pegleg labels
                (contents = [energy], shape = number of events)
        bins = integer number of data points you want (range/bins = width)
        minenergy = minimum energy value to start cut at (default = 0.)
        maxenergy = maximum energy value to end cut at (default = 60.)
    Returns:
        Scatter plot with energy values on x axis (median of bin width)
        y axis has median of resolution with error bars containing 68% of resolution
    """
    if use_fraction:
        resolution = ((nn_reco-truth)/truth) # in fraction
    else:
        resolution = (nn_reco-truth)
    percentile_in_peak = 68.27
    left_tail_percentile  = (100.-percentile_in_peak)/2
    right_tail_percentile = 100.-left_tail_percentile
    energy_ranges  = np.linspace(minenergy,maxenergy, num=bins)
    energy_centers = (energy_ranges[1:] + energy_ranges[:-1])/2.
    medians  = np.zeros(len(energy_centers))
    err_from = np.zeros(len(energy_centers))
    err_to   = np.zeros(len(energy_centers))
    if use_old_reco:
        if use_fraction:
            resolution_reco = ((old_reco-truth)/truth)
        else:
            resolution_reco = (old_reco-truth)
        medians_reco  = np.zeros(len(energy_centers))
        err_from_reco = np.zeros(len(energy_centers))
        err_to_reco   = np.zeros(len(energy_centers))
    for i in range(len(energy_ranges)-1):
        en_from = energy_ranges[i]
        en_to   = energy_ranges[i+1]
        cut = (truth >= en_from) & (truth < en_to)
        lower_lim = np.percentile(resolution[cut], left_tail_percentile)
        upper_lim = np.percentile(resolution[cut], right_tail_percentile)
        median = np.percentile(resolution[cut], 50.)
        medians[i] = median
        err_from[i] = lower_lim
        err_to[i] = upper_lim
        if use_old_reco:
            lower_lim_reco = np.percentile(resolution_reco[cut], left_tail_percentile)
            upper_lim_reco = np.percentile(resolution_reco[cut], right_tail_percentile)
            median_reco = np.percentile(resolution_reco[cut], 50.)
            medians_reco[i] = median_reco
            err_from_reco[i] = lower_lim_reco
            err_to_reco[i] = upper_lim_reco
    #plt.figure(figsize=(10,7))
    plt.errorbar(energy_centers, medians, yerr=[medians-err_from, err_to-medians], xerr=[ energy_centers-energy_ranges[:-1], energy_ranges[1:]-energy_centers ], capsize=5.0, fmt='o',label=label)
    if use_old_reco:
        plt.errorbar(energy_centers, medians_reco, yerr=[medians_reco-err_from_reco, err_to_reco-medians_reco], xerr=[ energy_centers-energy_ranges[:-1], energy_ranges[1:]-energy_centers ], capsize=5.0, fmt='o',label="Pegleg Reco")
        plt.legend(loc="upper center")
    plt.plot([minenergy,maxenergy], [0,0], color='k')
    plt.xlim(minenergy,maxenergy)
    plt.xlabel("Energy range (GeV)")
    if use_fraction:
        plt.ylabel("Fractional resolution (log E) \n (reco - truth)/truth")
    else:
        plt.ylabel("Resolution (log E) \n reco - truth")
    plt.title("Resolution energy dependence")
    savename = "EnergyResolutionSlices"
    if use_fraction:
        savename += "Frac"
    if use_old_reco:
        savename += "_CompareOldReco"
    if save == True:
        plt.savefig("%s%s.png"%(savefolder,savename))


hist_min = np.min(true)
hist_max = np.max(true)
plt.figure(figsize=(10,7))
plot_energy_slices(true2, preds2, use_fraction=True, bins=20, minenergy=hist_min, maxenergy=hist_max, label="GNN Gen2")
plot_energy_slices(true, preds, use_fraction=True, bins=20, minenergy=hist_min, maxenergy=hist_max, label="GNN IC86")
plot_energy_slices(true3, preds3, use_fraction=True, bins=20, minenergy=np.min(true3), maxenergy=np.max(true3), label="SplineMPE Gen2")
plt.legend()

savefolder = path
savename = "EnergyResolutionSlices"
savename += "Frac"
plt.savefig("%s%s.png"%(savefolder,savename))

#plot_energy_slices(true, preds, use_fraction=False, use_old_reco=True, old_reco=preds2, bins=20, minenergy=hist_min, maxenergy=hist_max, save=True, savefolder=path)


