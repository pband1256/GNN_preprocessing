#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/icetray-start
#METAPROJECT: /cvmfs/icecube.opensciencegrid.org/users/hieule/bld/

import numpy as np
import argparse
import pickle
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
parser.add_argument("-i", "--input",type=str,default="/mnt/scratch/lehieu1/data/21901/baseproc/**/*.i3.bz2",
                    dest="input_file", help="path and name of the input file")
parser.add_argument("-o", "--output",type=str,default="/mnt/home/lehieu1/IceCube/plot/iceprod/21901_hist",
                    dest="output_name", help="path and name for output file (without extension)")
parser.add_argument("-g", "--gcdfile",type=str,default="/cvmfs/icecube.opensciencegrid.org/users/gen2-optical-sim/gcd/IceCubeHEX_Sunflower_240m_v3.2.2_ExtendedDepthRange_mDOM.GCD.i3.bz2",dest='gcdfile',help="path for GCD file")
parser.add_argument("--bins",type=int,default=20,dest='bins')
parser.add_argument("--reco_dict",type=str,default=None,dest='reco_dict')

args = parser.parse_args()
infile = args.input_file + "/*.i3.bz2"
outfile = args.output_name
gcdfile = dataio.I3File(args.gcdfile)

def read_files(filename_list, gcdfile=gcdfile):
    # Reading I3MCTree zenith & energy and SplineMPE reco quantities
    true_zenith = np.empty(0); true_azimuth = np.empty(0); true_energy = np.empty(0)
    reco_zenith = np.empty(0); reco_azimuth = np.empty(0); reco_energy = np.empty(0)

    # Keep track of which indices in the truth array don't have SplineMPE reco
    for event_file_name in filename_list:
        print("reading file: {}".format(event_file_name))
        event_file = dataio.I3File(event_file_name)

        while event_file.more():
            try:
                frame = event_file.pop_physics()
            except:
                continue

            # Skip IC86 frames
            if frame['I3EventHeader'].sub_event_stream == 'IC86_SMT8':
                print("Skipping IC86 frame.")
                continue

            try: 
                # Reco values
                reco = frame['SplineMPEMuEXDifferential']
                reco_zenith  = np.append(reco_zenith,  reco.dir.zenith)
                reco_azimuth = np.append(reco_azimuth, reco.dir.azimuth)
                reco_energy  = np.append(reco_energy,  np.log10(reco.energy))
            except:
                #print("No SplineMPE reco frame found.")
                reco_zenith  = np.append(reco_zenith,  np.nan)
                reco_azimuth = np.append(reco_azimuth, np.nan)
                reco_energy  = np.append(reco_energy,  np.nan)
                #continue

            # Truth values
            # nu.pos.x, nu.pos.y, nu.pos.z, nu.dir.zenith, nu.dir.azimuth
            # nu.energy, nu.time
            event = frame["I3MCTree"]
            nu = event[0]
            true_zenith  = np.append(true_zenith,  nu.dir.zenith)
            true_azimuth = np.append(true_azimuth, nu.dir.azimuth)
            true_energy  = np.append(true_energy,  np.log10(nu.energy))

        # close the input file once we are done
        del event_file

    return {"true_zenith": true_zenith, "true_azimuth": true_azimuth, "true_energy": true_energy,
            "reco_zenith": reco_zenith, "reco_azimuth": reco_azimuth, "reco_energy": reco_energy}

def hist_i3(file_names, reco_dict, gcdfile=gcdfile, bins=args.bins, plotfile=outfile):

    zenith = reco_dict['true_zenith']
    azimuth = reco_dict['true_azimuth']
    energy = reco_dict['true_energy']

    fig = plt.figure(figsize=(22, 6))
    axE = fig.add_subplot(131)
    axZ = fig.add_subplot(132)
    axA = fig.add_subplot(133)

    hist_min = np.min(energy)
    hist_max = np.max(energy)
    #logbins = np.logspace(np.log10(hist_min), np.log10(hist_max), bins)
    linbins = np.linspace(hist_min, hist_max, bins)

    axE.hist(energy, bins=linbins, alpha=0.5)
    axE.set_yscale("log")
    axZ.hist(np.cos(np.deg2rad(zenith)), bins=bins, alpha=0.5, label="Visible")
    axA.hist(azimuth, bins=bins, alpha=0.5, label="Visible")
    #axE.hist(energy_m, bins=linbins, alpha=0.5)
    #axZ.hist(np.cos(np.deg2rad(zenith_m)), bins=bins, alpha=0.5, label="Not visible")
    #axA.hist(azimuth_m, bins=bins, alpha=0.5, label="Not visible")
    #axZ.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #      fancybox=True, shadow=True, ncol=2, fontsize=font)

    font = 20
    axE.set_ylabel(r"Counts", fontsize=font)
    axE.set_xlabel(r"$\log_{10} E_{\mu}$ [GeV]", fontsize=font)
    axZ.set_xlabel(r"$\cos \theta$", fontsize=font)
    axA.set_xlabel(r"$\phi$ [deg]", fontsize=font)
    axE.tick_params(axis='both', labelsize=font)
    axZ.tick_params(axis='both', labelsize=font)
    axA.tick_params(axis='both', labelsize=font)

    plotfile += '.png'
    print("Saving plot to " + plotfile + "...\n")
    plt.tight_layout()
    plt.savefig(plotfile) #, bbox_inches='tight', pad_inches=0.05)
    plt.clf()

def reco_i3(file_names, outfile=outfile):
    outfile += '.pkl'
    print("Saving dict to "+ outfile + "...\n")
    reco_dict = read_files(file_names, gcdfile)
    with open(outfile, 'wb') as handle:
        pickle.dump(reco_dict, handle)
    return reco_dict

#Construct list of filenames
import glob

file_names = sorted(glob.glob(infile))

if args.reco_dict is None:
    print("Compiling dictionary...")
    reco_dict = reco_i3(file_names)
    #Call function to read and label files
else:
    print("Loading dictionary from" + args.reco_dict + "...\n")
    with open(args.reco_dict, 'rb') as handle:
        reco_dict = pickle.load(handle)
print("Plotting histogram...\n")
hist_i3(file_names, reco_dict=reco_dict)
