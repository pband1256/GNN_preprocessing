from icecube import icetray
from I3Tray import I3Tray
from icecube.dataclasses import I3MapStringDouble
from icecube import dataclasses, dataio
import pandas as pd
import numpy as np

# Only filters P frames
# File output is not .zst compressed

preds = pd.read_csv('/mnt/home/lehieu1/IceCube/code/GNN/gnn_icecube/models/071420_20layers_5050/1/preds.csv', index_col=False)
track = True

data = preds[(preds.truth==float(track)) & (preds.prediction < 0.5) & (preds.prediction > 0.3)]
fileList = data.filename.unique()
print(fileList)

def check_id(frame, event_list):
  return frame["I3EventHeader"].event_id in event_list

for name in fileList:
  event_list = set(data.event_id[data.filename == name].sort_values())
  infile = name
  print(infile)
  outfile = '/mnt/home/lehieu1/track/processed_' + name.replace('/mnt/scratch/lehieu1/data/21002/0000000-0000999/','')

  tray = I3Tray()
  tray.AddModule('I3Reader', 'reader',
                 Filename=infile)
  # Filtering code
  tray.AddModule(check_id, event_list=event_list, Streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Physics])
  tray.AddModule("I3Writer", 'writer',
                 Filename=outfile)
  tray.Execute()
  tray.Finish()
