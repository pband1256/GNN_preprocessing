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
                    dest="output", help="path and name of output plot file")
parser.add_argument('-l','--lrate',type=float,nargs='+', default=[5e-3,5e-4,5e-5,5e-6],
                    dest="lrate",help='learning rates')
parser.add_argument('-n','--norm',type=int,default=0,
                    dest='norm',help='loss normalization flag')
# Use like this:
# python plot_GNN.py -l 5e-3 5e-4 5e-5 5e-6
parser.add_argument("--nb_train", type=str,default='',
                    dest="nb_train", help="number of training samples")
parser.add_argument("--nb_val", type=str,default='',
                    dest="nb_val", help="number of validation samples")
parser.add_argument("--batch_size", type=str,default='',
                    dest="batch_size", help="batch size")
parser.add_argument("--nb_hidden", type=str,default='',
                    dest="nb_hidden", help="number of hidden units")
parser.add_argument("--nb_layer", type=str,default='',
                    dest="nb_layer", help="number of convolutional layers")
args = parser.parse_args()


def loss_plot(ax,data,loss_type='train',lrate=args.lrate,normalize=args.norm):
    if loss_type == 'train':
        loss_type = 'train_loss'
        ax.set_title("Training loss")
    elif loss_type == 'val' or loss_type == 'validation':
        loss_type = 'val_loss'
        ax.set_title("Validation loss")
    if normalize:
        data[loss_type] = data[loss_type]/max(data[loss_type])
    ax.plot(data['Epoch'],data[loss_type])
    ax.axvline(x=0,linewidth=1, color='r')
    ax.annotate(s='lrate='+str(lrate[0]),xy=(1,max(data[loss_type])),rotation=90)
    for i in range(1,len(lrate)):
        cutoff = (data['lrate']==lrate[i-1])[::-1].idxmax()
        ax.axvline(x=cutoff+1,linewidth=1, color='r')
        ax.annotate(s='lrate='+str(lrate[i]),xy=(cutoff+2,max(data[loss_type])),rotation=90)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
        
data = pd.read_csv(args.input)

fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(121)
loss_plot(ax,data,loss_type='train')
ax1 = fig.add_subplot(122)
loss_plot(ax1,data,loss_type='val')
fig.suptitle('Model with training = '+args.nb_train+', val = '+args.nb_val+', batch = '+args.batch_size+', hidden units = '+args.nb_hidden+', layers = '+args.nb_layer)
fig.savefig(args.output)
