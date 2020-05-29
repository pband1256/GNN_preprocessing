import pickle
import matplotlib.pyplot as plt
import numpy as np

filepath = '/mnt/scratch/lehieu1/training_files/processed/'
filename = ['weight1_energy/train_file.pkl','500GeV_min_cuts_withenergy/train_file.pkl']
label = ['No cuts', 'min. 500GeV track cuts']
name = 'training_hist.png'
title = 'Training set histograms'

def plt_loghist(x, ax, bins, alpha=1,label=''):
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    ax.hist(x, bins=logbins, alpha=alpha, label=label)
    ax.set_xscale('log')

fig,ax = plt.subplots(figsize=(7,5))

for i in range(len(filename)):
    with open(filepath+filename[i],'rb') as f:
        E = pickle.load(f)[5]
        plt_loghist(E,ax,bins=100,alpha=0.8,label=label[i])
        
ax.axvline(500,c='red',linewidth=1)
ax.annotate(s='E=500GeV',xy=(600,400000),rotation=90,verticalalignment='top')
ax.legend()
ax.set_xlabel('Energy [GeV]')
ax.set_title(title)
fig.savefig('/mnt/home/lehieu1/IceCube/plot/'+name)
