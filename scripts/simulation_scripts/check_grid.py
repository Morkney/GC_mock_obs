import numpy as np
import os, re
import subprocess
import datetime
from file_read_backwards import FileReadBackwards
from config import *

# This script displays the status of Nbody6df simulations on this project.

display = False

def natural_key(string_):
  return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

# Find each simulation directory in the target suite:
#---------------------------------------------------------
sims = np.array(os.listdir(path + f'/Nbody6_sims/{suite}/'))
sims = sorted(sims[[i[-1].isdigit() for i in sims]])
#---------------------------------------------------------

# Go through each simulation directory:
#---------------------------------------------------------
for sim in sims:

  try:
    # Find all the 'run' subdirectories:
    #---------------------------------------------------------
    runs = []
    for root, dirs, files in os.walk(path + f'/Nbody6_sims/{suite}/{sim}'):
      if 'OUT3' in files:
        runs.append(root)
    runs = sorted(runs, key=natural_key)
    if 'run1' not in runs[0]:
      runs = runs[1:] + [runs[0]]
    run = runs[-1]
    #---------------------------------------------------------

    # Find the time since the err log file was last updated:
    #---------------------------------------------------------
    err = os.path.getctime(path + f'/Nbody6_sims/{suite}/{sim}/err')
    mod_time = (datetime.datetime.now() - datetime.datetime.fromtimestamp(err)).seconds
    #---------------------------------------------------------

    # Read the ends of the err and out log files:
    #---------------------------------------------------------
    err_end = str(subprocess.check_output(['tail', '-10', run+'/err'])[:-1])
    count = 0
    with FileReadBackwards(path + f'/Nbody6_sims/{suite}/{sim}/out') as file:
      for line in file:
        count += 1
        if 'ADJUST' in line:
          out_end = line.split('TIME =')[1][:5]
          break
        if count >= 10000:
          out_end = 'NaN'
          break
    #---------------------------------------------------------
  except:
    continue
    err = np.nan
    mod_time = 0.0
    err_end = 'VOID'
    out_end = 'VOID'

  # Create coloured strings corresponding to different scenarios:
  #---------------------------------------------------------
  pad1 = ' ' * (26 - len(sim))
  pad2 = ' ' * (8 - len(out_end))
  if err_end == 'VOID':
    colour = u'\u001b[38;5;8m'
  elif ('IEEE_' in err_end):
    colour = u'\u001b[38;5;160m'
  elif ('Closing' in err_end) or ('run' in run):
    colour = u'\u001b[34m'
    err_end = 'DONE'
  elif ('!!!overflow' in err_end) and (mod_time < 1800):
    colour = u'\u001b[38;5;208m'
  elif (mod_time >= 1800) or ('gpupot' not in err_end):
    colour = u'\u001b[38;5;160m'
  elif 'gpupot:' in err_end:
    colour = u'\u001b[38;5;046m'
  else:
    colour = u'\u001b[38;5;232m'
  #---------------------------------------------------------

  # Read the host server from run.log:
  #---------------------------------------------------------
  with open(path + f'/Nbody6_sims/{suite}/{sim}/run.log') as file:
    line = file.readlines()
    server = line[1].split(', ')[0]
  #---------------------------------------------------------

  # Write status data to screen:
  #---------------------------------------------------------
  if display or err_end != 'DONE':
    print('%s, %s: %s Last time: %s %s last err: %s%s%s' % \
          (server, sim, pad1, out_end, pad2, colour, err_end.split('\n')[-1], u'\033[0m'))
  #---------------------------------------------------------
#---------------------------------------------------------
