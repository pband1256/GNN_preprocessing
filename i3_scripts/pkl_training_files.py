#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/icetray-start
#METAPROJECT: combo/V00-00-03

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
                    dest="output_name",help="path and name for output file")
parser.add_argument("-g", "--gcdfile",type=str,default="/mnt/research/IceCube/gcd_file/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz",dest='gcdfile',help="path for GCD file")
parser.add_argument("--true_name",type=str,default=None,
                    dest="true_name", help="Name of key for true particle info if you want to check with I3MCTree[0]")

args = parser.parse_args()
input_file = args.input_file
output_name = args.output_name
gcdfile = dataio.I3File(args.gcdfile)
true_name = args.true_name

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

        # Pull coordinates for ALL DOMs
        StringLocationList.append((dom.position.x,dom.position.y,dom.position.z))
        OMKeyList.append((string,dom_id)) # goes from 1 to 86*60
    return np.array([OMKeyList, StringLocationList]) # size = [2,86*60]

def GetCoords(string, dom, StringLocations):
    return StringLocations[1][StringLocations[0].tolist().index((string,dom))]

def get_observable_features(frame,StringLocations,low_window=-500,high_window=20000):
    """
    Load observable features from IceCube files
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
    IC_strings = range(1,87)

    #Six summary variables: DOM x,y,z positions, sum first pulse, sum charges, time first pulse, 
    array_IC = np.zeros([len(IC_strings)*60,7]) # 86 strings, 60 DOMs each, [string, dom_index, position/charge & time summary]

    # Check if shifting trigger time is necessary

    # Collecting all time and charge in pulse list
    for omkey, pulselist in ice_pulses:
        dom_val = omkey.om          # indexed from 1 to 60
        string_index = omkey.string # indexed from 1 to 86
        dom_index = (string_index-1)*60 + (dom_val-1) # indexed from 0 to 85*60+59
        timelist = []
        chargelist = []

        # Each DOM/omkey can have multiple pulses, from first pulse to last pulse
        for pulse in pulselist:
            timelist.append(pulse.time)
            chargelist.append(pulse.charge)
                
        # Converting to np.array
        charge_array = np.array(chargelist)
        time_array = np.array(timelist)

        # Remove pulses so only those in certain time window are saved
#         original_num_pulses = len(timelist)
#         time_array_in_window = list(time_array)     # =timelist in this case
#         charge_array_in_window = list(charge_array) # =chargelist
#         for time_index in range(0,original_num_pulses):
#             time_value =  time_array[time_index]
#             # specified time window
#             if time_value < low_window or time_value > high_window:
#                 time_array_in_window.remove(time_value)
#                 charge_array_in_window.remove(charge_array[time_index])
#         charge_array = np.array(charge_array_in_window)
#         time_array = np.array(time_array_in_window)
        
        assert len(charge_array)==len(time_array), "Mismatched pulse time and charge"
        if len(charge_array) == 0:
            continue

        # Check that pulses are sorted in time
        for i_t,time in enumerate(time_array):
            assert time == sorted(time_array)[i_t], "Pulses are not pre-sorted!"
        
        coords = GetCoords(string_index,dom_val,StringLocations)
        # Charge weighted mean and stdev
        weighted_avg_time = np.average(time_array,weights=charge_array)
        weighted_std_time = np.sqrt(np.average((time_array - weighted_avg_time)**2, weights=charge_array))

        # Cartesian coordinates
        array_IC[dom_index,0] = coords[0]
        array_IC[dom_index,1] = coords[1]
        array_IC[dom_index,2] = coords[2]
        # Sum of charge of first pulse
        array_IC[dom_index,3] = np.sum(chargelist[0])
        # Sum of charge of all pulses
        array_IC[dom_index,4] = np.sum(chargelist)
        # Trigger time of first pulse
        # array_IC[dom_index,5] = time_array[0]
        # Charge weighted time mean
        array_IC[dom_index,5] = weighted_avg_time
        # Charge weighted time stdev
        array_IC[dom_index,6] = weighted_std_time
    array_IC = np.asarray(array_IC)
    
    # Check that pulse features are nonzero
    pulse_index = np.all(array_IC[:,3:6]!=0,axis=1)
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
    output_weights = []
    output_event_id = []
    output_filename = []
    output_energy = []
    output_num_pulses_per_dom = []
    isOther_count = 0  # check if not NC/CC
    
    StringLocations = GetStringLocations(gcdfile)

    for event_file_name in filename_list:
        print("reading file: {}".format(event_file_name))
        event_file = dataio.I3File(event_file_name)

        while event_file.more():
            try:
                frame = event_file.pop_physics()
            except:
                continue

            if frame["classification"] in [0,11]:
                print("Particle does not interact within the detector volume.")
                continue

            try:
                mu_energy = frame["mu_E_deposited"].value
            except:
                print("No clear muon found.")
                continue
            
            if frame["I3EventHeader"].sub_event_stream != "InIceSplit" and frame["I3EventHeader"].sub_event_stream != "Final":
                continue
            # ALWAYS USE EVENTS THAT PASSES CLEANING!
            #if use_cleaned_pulses:
#             try:
#                 cleaned = frame["SRTTWOfflinePulsesDC"]
#             except:
#                 continue

            # GET TRUTH LABELS
            try:
                event = frame["I3MCTree"]
            except:
                event = frame["I3MCTree_preMuonProp"] # x,y,z,ra,dec,time,energy,length
            nu = event[0]
            
            if (nu.type != dataclasses.I3Particle.NuMu and nu.type != dataclasses.I3Particle.NuMuBar\
                and nu.type != dataclasses.I3Particle.NuE and nu.type != dataclasses.I3Particle.NuEBar\
                and nu.type != dataclasses.I3Particle.NuTau and nu.type != dataclasses.I3Particle.NuTauBar):
                print("Particle is not neutrino.")
                continue           
 
            # All these are not necessary as of now
            nu_x = nu.pos.x
            nu_y = nu.pos.y
            nu_z = nu.pos.z
            nu_zenith = nu.dir.zenith
            nu_azimuth = nu.dir.azimuth
            nu_energy = nu.energy
            nu_time = nu.time
            isCC = frame['I3MCWeightDict']['InteractionType']==1.
            isNC = frame['I3MCWeightDict']['InteractionType']==2.
            isOther = not isCC and not isNC


            # input file sanity check: this should not print anything since "isOther" should always be false
            if isOther:
                print("isOTHER - not Track or Cascade...skipping event...")
                isOther_count += 1
                continue
            
            # set track classification for numu CC only
            if ((nu.type == dataclasses.I3Particle.NuMu or nu.type == dataclasses.I3Particle.NuMuBar) and isCC):
                isTrack = True
                isCascade = False
                if event[1].type == dataclasses.I3Particle.MuMinus or event[1].type == dataclasses.I3Particle.MuPlus:
                    track_length = event[1].length
                else:
                    #print("Second particle in MCTree not muon for numu CC? Skipping event...")
                    continue
            else:
                isTrack = False
                isCascade = True
                track_length = 0
            
            #Save flavor and particle type (anti or not)
            if (nu.type == dataclasses.I3Particle.NuMu):
                neutrino_type = 14
                particle_type = 0 #particle
            elif (nu.type == dataclasses.I3Particle.NuMuBar):
                neutrino_type = 14
                particle_type = 1 #antiparticle
            elif (nu.type == dataclasses.I3Particle.NuE):
                neutrino_type = 12
                particle_type = 0 #particle
            elif (nu.type == dataclasses.I3Particle.NuEBar):
                neutrino_type = 12
                particle_type = 1 #antiparticle
            elif (nu.type == dataclasses.I3Particle.NuTau):
                neutrino_type = 16
                particle_type = 0 #particle
            elif (nu.type == dataclasses.I3Particle.NuTauBar):
                neutrino_type = 16
                particle_type = 1 #antiparticle
            else:
                print("Do not know first particle type in MCTree, should be neutrino, skipping this event")
                continue
            
            IC_array = get_observable_features(frame,StringLocations)

            # regression variables
            # OUTPUT: [ nu energy, nu zenith, nu azimuth, nu time, nu x, nu y, nu z, track length (0 for cascade), isTrack, flavor, type (anti = 1), isCC]
            
#             output_labels.append( np.array([ float(nu_energy), float(nu_zenith), float(nu_azimuth), float(nu_time), float(nu_x), float(nu_y), float(nu_z), float(track_length), float(isTrack), float(neutrino_type), float(particle_type), float(isCC) ]) )

            
            output_labels.append(float(isTrack)) # label y
            output_features_IC.append(IC_array) # feature X, but x,y,z=string,DOM,DOM_index as of now
            output_weights.append(frame['I3MCWeightDict']['OneWeight'])
            output_event_id.append(frame["I3EventHeader"].event_id)
            output_filename.append(event_file_name)
            output_energy.append(mu_energy)    

        print("Got rid of %i events classified as other so far"%isOther_count)

        # close the input file once we are done
        del event_file

    X = np.asarray(output_features_IC)
    y = np.asarray(output_labels)
    weights = np.asarray(output_weights)
    event_id = np.asarray(output_event_id)
    filename = np.asarray(output_filename)
    energy = np.asarray(output_energy)

    return X, y, weights, event_id, filename, energy

#Construct list of filenames
import glob

file_name = input_file

event_file_names = sorted(glob.glob(file_name))
assert event_file_names,"No files loaded, please check path."

#Call function to read and label files
#features_IC, labels, num_pulses_per_dom, = read_files(event_file_names)
X, y, w, e, f, E = read_files(event_file_names,gcdfile)
full_data = [X,y,w,e,f,E]

#Save output to pickle file
import pickle
output_path = output_name
f = open(output_path, "wb")
pickle.dump(full_data,f)
f.close()
