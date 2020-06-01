import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",type=str,
                    dest="input", help="path and name of input .csv data")
parser.add_argument("-o", "--output",type=str,default='/mnt/home/lehieu1/IceCube/plot/GNN/'+datetime.now().strftime("%H%M%S_%m%d%Y"),
                    dest="output",help="path and name of output plot file")
parser.add_argument('-l','--lrate',type=float,nargs='+', default=[5e-3,5e-4,5e-5,5e-6],
                    dest="lrate",help='learning rates')
parser.add_argument('-t','--title',type=str,default='',
                    dest='title',help='plot title')
parser.add_argument('-m','--multi',type=int,default=1,
                    dest="multi",help='separate training and validation loss plots')
# Use like this:
# python plot_GNN.py -l 5e-3 5e-4 5e-5 5e-6
args = parser.parse_args()


def loss_plot(ax,data,loss_type='train',multi=1,lrate=args.lrate):
    if loss_type == 'train':
        loss_type = 'train_loss'
        ax.set_title("Training loss")
    elif loss_type == 'val' or loss_type == 'validation':
        loss_type = 'val_loss'
        ax.set_title("Validation loss")

    ax.plot(data['Epoch'],data[loss_type])
    ax.axvline(x=0,linewidth=1, color='r')
    if multi:
        ax.annotate(s='lrate='+str(lrate[0]),xy=(1,max(data[loss_type])),rotation=90,verticalalignment='top')
    else:
        ax.set_title('')
    for i in range(1,len(lrate)):
        cutoff = (data['lrate']==lrate[i-1])[::-1].idxmax()
        ax.axvline(x=cutoff+1,linewidth=1, color='r')
        if multi:
            ax.annotate(s='lrate='+str(lrate[i]),xy=(cutoff+2,max(data[loss_type])),rotation=90,verticalalignment='top')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
        
data = pd.read_csv(args.input)


if args.multi==0:
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111)
    loss_plot(ax,data,loss_type='train')
    loss_plot(ax,data,loss_type='val',multi=0)
    ax.legend()
else:
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(121)
    loss_plot(ax,data,loss_type='train')
    ax1 = fig.add_subplot(122)
    loss_plot(ax1,data,loss_type='val')

fig.suptitle(args.title)
fig.savefig(args.output)
