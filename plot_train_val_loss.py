import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",type=str,
                    dest="input", help="model name")
parser.add_argument("-o", "--output",type=str,default='/mnt/home/lehieu1/IceCube/plot/GNN/'+datetime.now().strftime("%H%M%S_%m%d%Y"),
                    dest="output",help="name of output plot file")
parser.add_argument('-t','--title',type=str,default='',
                    dest='title',help='plot title')
parser.add_argument('--nb_files',type=str,default=1,
                    dest='nb_files',help='number of (multi) files')
args = parser.parse_args()

def best_stats(path,sets=[0,1,2]):
    train_AUC = []
    val_AUC = []
    test_AUC = []
    for i in sets:
        data = pd.read_csv(path+'/'+str(i)+"/training_stats.csv")
        with open(path+'/'+str(i)+"/best_scores.yml", 'r') as stream:
            try:
                ind = yaml.safe_load(stream)['epoch']
            except yaml.YAMLError as exc:
                print(exc)
        try:
            with open(path+'/'+str(i)+"/test_scores.yml", 'r') as stream:
                try:
                    test_AUC.append(yaml.safe_load(stream)['roc auc'])
                except yaml.YAMLError as exc:
                    print(exc)
        except (IOError, OSError):
            print('Test file {} not found. Model might still have more epochs to run.'.format(i))
            test_AUC.append(float('NaN'))
        train_AUC.append(data['train_roc'].tolist()[ind])
        val_AUC.append(data['val_roc'].tolist()[ind])
    return train_AUC,val_AUC,test_AUC

def loss_plot(ax,path,sets=[0,1,2],nb_files=1):
    if len(ax) != len(sets):
        raise Exception("Number of figures and number of data sets are not the same.")
    elif len(ax)==0 or len(sets)==0:
        raise Exception("Must specify at least one axis or data set.")
        
    # Offsetting annotation labels from lines
    annotate_offset = 2
    
    for i in range(len(sets)):
        data = pd.read_csv(path+'/'+str(sets[i])+'/training_stats.csv')
        # Getting list of learning rate and sort small to large
        lrate = np.sort(np.unique(data['lrate']))[::-1]
        epochs = data['Epoch'][::nb_files]/nb_files

        # Plotting loss every nb_files epoch
        ax[i].plot(epochs,data['train_loss'][::nb_files])
        ax[i].plot(epochs,data['val_loss'][::nb_files])
        
        # Learning rate lines
        max_loss = max(np.concatenate([data['val_loss'],data['train_loss']]))
        ax[i].axvline(x=0,linewidth=1, color='r')
        ax[i].annotate(s='lrate='+str("{:.0e}".format(lrate[0])),xy=(annotate_offset,max_loss),rotation=90,verticalalignment='top')
        for rate in range(1,len(lrate)):
            cutoff = (data['lrate']==lrate[rate-1])[::-1].idxmax()/nb_files
            ax[i].axvline(x=cutoff,linewidth=1, color='r')
            ax[i].annotate(s='lrate='+str("{:.0e}".format(lrate[rate])),xy=(cutoff+annotate_offset,max_loss),rotation=90,verticalalignment='top')
        ax[i].set_xlabel("Epoch (full pass)")
        ax[i].set_ylabel("Loss")
    
        # Best model line
        with open(path+'/'+str(i)+"/best_scores.yml", 'r') as stream:
            try:
                best_epoch = yaml.safe_load(stream)['epoch']/nb_files
            except yaml.YAMLError as exc:
                print(exc)
        ax[i].axvline(x=best_epoch,linewidth=1, color='g')
        ax[i].annotate(s='best model',xy=(best_epoch+annotate_offset,max_loss),rotation=90,verticalalignment='top')
    
    # AUCs for best model box, criteria: val AUC
    train_AUC,val_AUC,test_AUC = best_stats(path,sets=sets)
    best_run = val_AUC.index(max(val_AUC))
    ax[-1].annotate('Best run: {}\ntrain AUC = {:.4}\nval AUC = {:.4}\ntest AUC = {:.4}'.format(sets[best_run],train_AUC[best_run],val_AUC[best_run],test_AUC[best_run]), 
                    xy=(0.75, 0.5), xycoords="axes fraction",
                    va="center", ha="left", bbox=dict(fc="w"))
    
    # Legend box
    handles, _ = ax[0].get_legend_handles_labels()
    labels = ['Training loss', 'Validation loss']
    ax[0].legend(handles, labels,ncol=2,loc='center',bbox_to_anchor=(0.5, 1),fancybox=True,shadow=True)
        
dataset = "061020_5050_normaltest_10x100k_20patience"
extra = ''

path = '/home/jovyan/GNN/gnn_icecube/models/'+dataset

with open('/home/jovyan/GNN/gnn_icecube/models/'+dataset+"/1/args.yml", 'r') as stream:
    try:
        args = yaml.load(stream, Loader=yaml.Loader)
        nb_train = args.nb_train
        nb_val = args.nb_val
        nb_test = args.nb_test
        nb_hidden = args.nb_hidden
        nb_layer = args.nb_layer
        batch_size = args.batch_size
    except yaml.YAMLError as exc:
        print(exc)

print("Current set: "+dataset)
fig, ax = plt.subplots(figsize=(20,5),nrows=1,ncols=3,sharey=True)
loss_plot(ax,path,sets=[0,1,2],nb_files=10)
suptitle = 'Model w/ training='+str(nb_train)+', val='+str(nb_val)+', batch='+str(batch_size)+', hidden units='\
             +str(nb_hidden)+', layers='+str(nb_layer)+'\nModel name: '+dataset
fig.suptitle(suptitle,linespacing = 1.5)
plt.show()
fig.savefig('./plot/'+dataset+'.png')
