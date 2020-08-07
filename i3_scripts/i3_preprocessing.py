#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/icetray-start
#METAPROJECT: combo/V00-00-03

from I3Tray import *
from icecube import icetray, dataclasses, dataio, MuonGun, phys_services
from icecube.simprod import segments

import argparse
import numpy as np
import glob
import reco_utils as utils

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name",type=str,required=True,nargs="+",
                    dest="name", help="path and name of input files")
parser.add_argument("--gcdfile",type=str,default="/mnt/research/IceCube/gcd_file/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz",
                    dest="gcdfile", help="path of GCD file")
args = parser.parse_args()


#rename MMCTrackList to something temporary
def I3MCTpmp_2_I3MCT(frame):
    frame["MMCTrackList_temp"] = frame["MMCTrackList"]
    del frame["MMCTrackList"]

#delete "new" mmctracklist and replace with the temp (temp already had coincident and primary events mereged and propagated)
def I3MCT_2_I3MCTpmp(frame):
    del frame["MMCTrackList"]
    frame["MMCTrackList"] = frame["MMCTrackList_temp"]
    del frame["MMCTrackList_temp"]

#repropagate the muons in mctree
randomService = phys_services.I3SPRNGRandomService(
    seed = 10000,
    nstreams = 200000000,
    streamnum = 100014318)


for name in args.name:
    infile = name
    outfile = name.replace('Level2_IC86','processed/processed_Level2_IC86')

    # 1. Generate I3MCTree object from I3MCTree_preMuonProp with seed
    # 2. Calculate muon entry and deposited energy
    # 3. Insert event label

    tray = I3Tray()
    tray.AddModule('I3Reader', 'reader',
	     Filename=infile)
    print("--------- Propagating muon and generating I3MCTree ----------")
    tray.Add(I3MCTpmp_2_I3MCT,Streams=[icetray.I3Frame.DAQ,icetray.I3Frame.Physics])
    tray.Add(segments.PropagateMuons, 'PropagateMuons',
	     RandomService=randomService,
	     SaveState=True,
	     InputMCTreeName="I3MCTree_preMuonProp",
	     OutputMCTreeName="I3MCTree")
    tray.Add(I3MCT_2_I3MCTpmp,Streams=[icetray.I3Frame.DAQ,icetray.I3Frame.Physics])

    print("--------- Calculating muon energy -----------")
    tray.AddModule(utils.get_most_E_muon_info, gcdfile=args.gcdfile)
    print("--------- Classifying event ----------")
    tray.AddModule(utils.classify, gcdfile=args.gcdfile)

    tray.AddModule("I3Writer", 'writer',
	     Filename=outfile)
    tray.Execute()
    tray.Finish()
