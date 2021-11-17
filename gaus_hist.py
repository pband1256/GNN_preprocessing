#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/icetray-start
#METAPROJECT: /cvmfs/icecube.opensciencegrid.org/users/hieule/bld/

import numpy as np
import argparse

from icecube import icetray, dataio, dataclasses, MuonGun
import icecube.MuonGun
from icecube.weighting.weighting import from_simprod
from I3Tray import I3Units

from collections import OrderedDict
import itertools
import random
import matplotlib.pyplot as plt

def GetStringLocations(gcdfile):
    frame = gcdfile.pop_frame() # pop I frame
    frame = gcdfile.pop_frame() # pop G frame
    geometry = frame['I3Geometry']
    StringLocationList = []
    OMKeyList = []

    for omkey in geometry.omgeo.keys():
        dom = geometry.omgeo[omkey]
        string = omkey.string
        dom_id = omkey.om
        pmt_id = omkey.pmt

        # Pull coordinates for ALL DOMs and directions for PMTs
        StringLocationList.append((dom.position.x, dom.position.y, dom.position.z))
        OMKeyList.append((string, dom_id, pmt_id)) # goes from 1-86*60+122*80 + 24 pmts for upgrade
    return np.array([OMKeyList, StringLocationList]) # size = [2,86*60+122*80]

def GetCoords(StringLocationList, string, dom, pmt=0):
    # return [x, y, z, zen, azi]
    return StringLocationList[1][StringLocationList[0].tolist().index([string, dom, pmt])]

gcdfile = "/cvmfs/icecube.opensciencegrid.org/users/gen2-optical-sim/gcd/IceCubeHEX_Sunflower_240m_v3.2.2_ExtendedDepthRange_mDOM.GCD.i3.bz2"
gcdfile = dataio.I3File(gcdfile)
dom_id = 65         # indexed from 1 to 60 or 1 to 80
string_id = 1024    # indexed from 1 to 86 or 1001 to 1122
pmt_id = 12         # indexed from 0 to 23
StringLocations = GetStringLocations(gcdfile)

loc = GetCoords(StringLocations, string_id, dom_id, pmt_id)
emb = StringLocations[1]

# Gaussian kernel
# sigma = np.random.rand(1) * 0.02 + 0.99
sigma = 1000
# loc.size() = (3,)
# emb.size() = (nb_pts, 3)
emb = np.linalg.norm(np.repeat(np.expand_dims(loc,1), np.shape(emb)[0], axis=1).transpose() - emb, axis=1)

fig = plt.figure(figsize=(9, 6)) 
ax = fig.add_subplot(111)
ax.hist(emb, bins=50)
font = 20
ax.set_ylabel(r"Counts", fontsize=font)
ax.set_xlabel(r"Pairwise distance", fontsize=font)
ax.tick_params(axis='both', labelsize=font)
plt.tight_layout()
plt.savefig('/mnt/home/lehieu1/IceCube/code/GNN/gaus_plot.png')
plt.clf()

# emb.size() = (nb_pts,)
emb = np.exp(-emb**2 / sigma**2)

fig = plt.figure(figsize=(9, 6)) 
ax = fig.add_subplot(111)
n, bins, _ = ax.hist(emb, bins=50)
print(n)
font = 20
ax.set_ylabel(r"Counts", fontsize=font)
ax.set_xlabel(r"Gaussianed distance", fontsize=font)
ax.tick_params(axis='both', labelsize=font)
ax.set_yscale('log')
ax.set_title(r'$\sigma = {}$'.format(sigma), fontsize=font)
plt.tight_layout()
plt.savefig('/mnt/home/lehieu1/IceCube/code/GNN/gaus_plot_sig{}_1.png'.format(sigma))

# Softmax
emb = np.divide(np.exp(emb), np.sum(np.exp(emb)))

fig = plt.figure(figsize=(9, 6)) 
ax = fig.add_subplot(111)
n, bins, _ = ax.hist(emb, bins=50)
print(n)
font = 20
ax.set_ylabel(r"Counts", fontsize=font)
ax.set_xlabel(r"Softmaxed distance", fontsize=font)
ax.ticklabel_format(axis='x',style='sci', scilimits=(0,0))
ax.tick_params(axis='both', labelsize=font)
ax.set_yscale('log')
ax.set_title(r'$\sigma = {}$'.format(sigma), fontsize=font)
plt.tight_layout()
plt.savefig('/mnt/home/lehieu1/IceCube/code/GNN/gaus_plot_sig{}_2.png'.format(sigma))
