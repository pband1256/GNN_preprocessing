#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/icetray-start
#METAPROJECT: /cvmfs/icecube.opensciencegrid.org/users/hieule/bld/

import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from icecube import icetray, dataio, dataclasses, MuonGun
import icecube.MuonGun
from icecube.weighting.weighting import from_simprod
from I3Tray import I3Units

from collections import OrderedDict
import itertools
import random

## Create ability to change settings from terminal ##
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",type=str,default="/mnt/scratch/lehieu1/data/iceprod/baseproc/Sunflower_240m_mDOM_2.2x_MuonGun.021891.000010_baseproc.i3.bz2",
                    dest="input_file", help="path and name of the input file")
parser.add_argument("-o", "--output",type=str,default="/mnt/home/lehieu1/IceCube/plot/iceprod/i3_hist_010.png",
                    dest="output_name", help="path and name for output file")
parser.add_argument("-g", "--gcdfile",type=str,default="/cvmfs/icecube.opensciencegrid.org/users/gen2-optical-sim/gcd/IceCubeHEX_Sunflower_240m_v3.2_ExtendedDepthRange_mDOM.GCD.i3.bz2",dest='gcdfile',help="path for GCD file")
parser.add_argument("-b", "--bins",type=int,default=20,dest='bins')


args = parser.parse_args()
input_file = args.input_file
plotfile = args.output_name
gcdfile = dataio.I3File(args.gcdfile)

def read_files(filename_list, gcdfile):
    zenith = np.empty(0)
    energy = np.empty(0)

    for event_file_name in filename_list:
        print("reading file: {}".format(event_file_name))
        event_file = dataio.I3File(event_file_name)

        while event_file.more():
            try:
                frame = event_file.pop_physics()
            except:
                continue

            # GET TRUTH LABELS
            event = frame["I3MCTree"]
            nu = event[0]
            #nu_x = nu.pos.x
            #nu_y = nu.pos.y
            #nu_z = nu.pos.z
            nu_zenith = nu.dir.zenith
            #nu_azimuth = nu.dir.azimuth
            nu_energy = nu.energy
            #nu_time = nu.time

            zenith = np.append(zenith, nu_zenith)
            energy = np.append(energy, nu_energy)

        # close the input file once we are done
        del event_file

    return zenith, energy

#Construct list of filenames
import glob

file_names = np.empty(0)
for file_name in sorted(glob.glob(input_file)):
    file_names = np.append(file_names, file_name)

#Call function to read and label files
zenith, energy = read_files(file_names, gcdfile)
print(np.shape(energy))

fig = plt.figure(figsize=(17, 6))
axE = fig.add_subplot(121)
axZ = fig.add_subplot(122)

hist_min = np.min(energy)
hist_max = np.max(energy)
bins = np.logspace(np.log10(hist_min), np.log10(hist_max), args.bins)

axE.hist(energy, bins=bins)
axE.set_xscale("log")
axZ.hist(np.rad2deg(zenith), bins=args.bins)

axE.set_xlabel("Energy (GeV)")
axZ.set_xlabel("Zenith (deg)")

print(plotfile)
plt.savefig(plotfile)
plt.clf()
