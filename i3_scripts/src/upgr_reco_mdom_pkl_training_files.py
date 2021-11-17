#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/icetray-start
#METAPROJECT: /cvmfs/icecube.opensciencegrid.org/users/hieule/bld/

#############################
# Read IceCube files and create training file (pickle)
#   Modified from code written by Claudio Kopper and Jessie Micallef
#   get_observable_features = access data from IceCube files
#   read_files = read in files and add truth labels
#   Can take 1 or multiple files
#   Input:
#       -i input: name of input file, include path
#       -n name: name for output file, automatically puts in my scratch
#       -r reco: flag to save Level5p pegleg reco output (to compare)
#       --emax: maximum energy saved (60 is default, so keep all < 60 GeV)
#       --cleaned: if you want to pull from SRTTWOfflinePulsesDC, instead of SplitInIcePulses
#       --true_name: string of key to check true particle info from against I3MCTree[0] 
##############################

import numpy as np
import argparse

from icecube import icetray, dataio, dataclasses, MuonGun
import icecube.MuonGun
from icecube.weighting.weighting import from_simprod
from I3Tray import I3Units

from collections import OrderedDict
import itertools
import random

## Create ability to change settings from terminal ##
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",type=str,
                    dest="input_file", help="path and name of the input file")
parser.add_argument("-o", "--output",type=str,
                    dest="output_name", help="path and name for output file")
parser.add_argument("-g", "--gcdfile",type=str,default="/cvmfs/icecube.opensciencegrid.org/users/gen2-optical-sim/gcd/IceCubeHEX_Sunflower_240m_v3.2.2_ExtendedDepthRange_mDOM.GCD.i3.bz2",dest='gcdfile',help="path for GCD file")
parser.add_argument("--IC86", help="flag for IC86 strings only", action='store_true')

args = parser.parse_args()
input_file = args.input_file
output_name = args.output_name
IC86 = args.IC86
gcdfile = dataio.I3File(args.gcdfile)

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
        StringLocationList.append((dom.position.x, dom.position.y, dom.position.z, dom.orientation.dir_zenith, dom.orientation.dir_azimuth))
        OMKeyList.append((string, dom_id, pmt_id)) # goes from 1-86*60+122*80*
    return np.array([OMKeyList, StringLocationList]) # size = [2,86*60+122*80]

def GetCoords(StringLocationList, string, dom, pmt=0):
    # return [x, y, z, zen, azi]
    return StringLocationList[1][StringLocationList[0].tolist().index((string, dom, pmt))]

def get_observable_features(frame,StringLocations,low_window=-500,high_window=20000):
    """

    Receives:
        frame = IceCube object type from files
    Returns:
        observable_features: Observables dictionary
    """
    try:
        ice_pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(frame,'SplitInIcePulses')
    except:
        ice_pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(frame,'InIceDSTPulses')

    #Look inside ice pulses and get stats on charges and time
    # IC 86  strings * 60 DOMs * 1  PMT  [string, dom_index, position/charge & time summary]
    IC_strings = 86
    # G2 122 strings * 80 DOMs * 24 PMTs
    G2_strings = 122
    array_IC = np.zeros([IC_strings*60*1+G2_strings*80*24, 10])

    # Collecting all time and charge in pulse list
    for omkey, pulselist in ice_pulses:
        dom_id = omkey.om        # indexed from 1 to 60 or 1    to 80
        string_id = omkey.string # indexed from 1 to 86 or 1001 to 1122
        pmt_id = omkey.pmt       # indexed from 0 to 23
        if string_id > 1000:
            dom_index = 86*60 + (string_id%1000-1)*80*24 + (dom_id-1) + pmt_id 
        else:
            dom_index = (string_id-1)*60+(dom_id-1) # indexed from 0 to 85*60+59
        timelist = []
        chargelist = []

        # Each DOM/omkey can have multiple pulses, from first pulse to last pulse
        for pulse in pulselist:
            timelist.append(pulse.time)
            chargelist.append(pulse.charge)
                
        # Converting to np.array
        charge_array = np.array(chargelist)
        time_array = np.array(timelist)
        
        assert len(charge_array)==len(time_array), "Mismatched pulse time and charge"
        if len(charge_array) == 0:
            continue

        # Check that pulses are sorted in time
        for i_t,time in enumerate(time_array):
            assert time == sorted(time_array)[i_t], "Pulses are not pre-sorted!"
        
        coords = GetCoords(StringLocations, string_id, dom_id, pmt_id)
        # Charge weighted mean and stdev
        weighted_avg_time = np.average(time_array,weights=charge_array)
        weighted_std_time = np.sqrt(np.average((time_array - weighted_avg_time)**2, weights=charge_array))

    # 9 summary variables: DOM x,y,z positions, PMT zen,azi dir, sum first pulse, sum charges, weighted time mean & stdev
    # 2 labels: energy, direction (zenith/azimuth or x/y/z)
        # Cartesian coordinates + spherical orientation
        array_IC[dom_index,0:5] = coords
        array_IC[dom_index,5] = np.sum(chargelist[0]) # Sum of charge of first pulse
        array_IC[dom_index,6] = np.sum(chargelist)    # Sum of charge of all pulses
        array_IC[dom_index,7] = time_array[0]         # Trigger time of first pulse
        array_IC[dom_index,8] = weighted_avg_time     # Charge weighted time mean
        array_IC[dom_index,9] = weighted_std_time     # Charge weighted time stdev
    array_IC = np.asarray(array_IC)
    
    # Check that pulse features are nonzero
    pulse_index = np.all(array_IC[:,5:7]!=0,axis=1)
    array_IC = array_IC[pulse_index]
    return array_IC

def read_files(filename_list, gcdfile):
    """
    Read list of files, make sure they pass L5 cuts, create truth labels
    Receives:
        filename_list = list of strings, filenames to read data from
    Returns:
        output_features_DC = dict with input observable features from the DC strings
        output_features_IC = dict with input observable features from the IC strings
        output_labels = dict with output labels (energy, zenith, azimuth, time, x, y, z, 
                        tracklength, isTrack, flavor ID, isAntiNeutrino, isCC)
        output_reco_labels = dict with PegLeg output labels (energy, zenith, azimith, time, x, y, z)
        output_initial_stats = array with info on number of pulses and sum of charge "inside" the strings used 
                                vs. "outside", i.e. the strings not used (pulse count outside, charge outside,
                                pulse count inside, charge inside) for finding statistics
        output_num_pulses_per_dom = array that only holds the number of pulses seen per DOM (finding statistics)
        output_trigger_times = list of trigger times for each event (used to shift raw pulse times)
    """
    output_features_IC = []
    output_labels = []
    output_event_id = []
    output_filename = []
    output_energy = []
    output_reco = []
    output_num_pulses_per_dom = []
    
    StringLocations = GetStringLocations(gcdfile)

    for event_file_name in filename_list:
        print("reading file: {}".format(event_file_name))
        event_file = dataio.I3File(event_file_name)

        while event_file.more():
            try:
                frame = event_file.pop_physics()
            except:
                continue

            #try:
            #    if frame["classification"] in [0,11]:
            #        print("Particle does not interact within the detector volume.")
            #        continue
            #except:
            #    print("No classification frame found, proceed normally.")


            if (frame["I3EventHeader"].sub_event_stream == "IC86_SMT8") and (IC86):
                print("IC86 P-frame found.")
            elif (frame["I3EventHeader"].sub_event_stream != "IC86_SMT8") and (not IC86):
                print("Gen2 P-frame found.")
            else:
                print("Not requested frame type, discarded.")
                continue

            # MUON INFORMATION
            mu = frame["MCMuon"]      # energy,zen,azi -> x,y,z
            mu_x = mu.pos.x           # vertex location
            mu_y = mu.pos.y
            mu_z = mu.pos.z
            mu_zenith = mu.dir.zenith # direction
            mu_azimuth = mu.dir.azimuth
            mu_energy = mu.energy     # energy

            # GET TRUTH LABELS
            event = frame["I3MCTree"] # x,y,z,ra,dec,time,energy,length
            nu = event[0]
            #nu_x = nu.pos.x
            #nu_y = nu.pos.y
            #nu_z = nu.pos.z
            #nu_zenith = nu.dir.zenith
            #nu_azimuth = nu.dir.azimuth
            #nu_energy = nu.energy
            #nu_time = nu.time

            # Regression labels
            energy = np.log10(mu_energy) # train with log10(E)
            zenith = mu_zenith
            azimuth = mu_azimuth
            def toCart(theta,phi):
                r = 1 # for unit vector
                x = r*np.cos(phi)*np.sin(theta)
                y = r*np.sin(phi)*np.sin(theta)
                z = r*np.cos(theta)
                return x,y,z
            x, y, z = toCart(zenith, azimuth)
            labels = np.array([energy, zenith, azimuth, x, y, z])

            IC_array = get_observable_features(frame,StringLocations)

            try:
                reco_energy = frame["SplineMPEMuEXDifferential"].energy
            except:
                reco_energy = float("NaN")
                #continue

            output_labels.append(labels) # label y
            output_features_IC.append(IC_array) # feature X, but x,y,z=string,DOM,DOM_index as of now
            output_event_id.append(frame["I3EventHeader"].event_id)
            output_filename.append(event_file_name)
            output_energy.append(mu_energy)
            output_reco.append(reco_energy)
        # close the input file once we are done
        del event_file

    # Feature array: DOM pos (x,y,z), PMT dir (zen,azi), summmary vars (sum pulses x2, charge weighted time x2)
    X = np.asarray(output_features_IC)
    # Truth labels: muon energy, direction (zen,azi/x,y,z)
    y = np.asarray(output_labels)
    #weights = np.asarray(output_weights)
    weights = np.ones(np.shape(y))
    event_id = np.asarray(output_event_id)
    filename = np.asarray(output_filename)
    # I3MCTree truth energy
    energy = np.asarray(output_energy)
    # SplineMPE reco energy for comparison and plotting
    reco = np.asarray(output_reco)

    return X, y, weights, event_id, filename, energy, reco

# Construct list of filenames
import glob

file_name = input_file

event_file_names = sorted(glob.glob(file_name))
assert event_file_names,"No files loaded, please check path."

# Call function to read and label files
# features_IC, labels, num_pulses_per_dom, = read_files(event_file_names)
X, y, w, e, f, E, r = read_files(event_file_names,gcdfile)
full_data = [X,y,w,e,f,E,r]

#Save output to pickle file
import pickle
output_path = output_name
f = open(output_path, "wb")
pickle.dump(full_data,f)
f.close()
