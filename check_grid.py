import numpy as np
import os, re
import subprocess
import datetime
from file_read_backwards import FileReadBackwards

display = False

def natural_key(string_):
  return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

# Find each simulation directory:
dirs = np.array(os.listdir('./'))
dirs = dirs[[i[0].isdigit() for i in dirs]]
dirs = dirs[np.argsort(([int(i[:-1]) for i in dirs]))[::-1]]

# Go through each simulation directory:
for resolution in dirs:
  print('')
  print(u'\u001b[34;1m%s\033[0m' % resolution)
  sims = np.sort(np.array(os.listdir('%s/' % resolution)))
  for sim in sims:

    if '_15' not in sim: continue

    # Find out how far the simulation has gotten:
    try:
      runs = []
      for root, dirs, files in os.walk('%s/%s' % (resolution, sim)):
        if 'OUT3' in files:
          runs.append(root)
      runs = sorted(runs, key=natural_key)
      if 'run1' not in runs[0]:
        runs = runs[1:] + [runs[0]]
      run = runs[-1]

      err = os.path.getctime(run+'/err')
      mod_time = (datetime.datetime.now() - datetime.datetime.fromtimestamp(err)).seconds

      err_end = subprocess.check_output(['tail', '-10', run+'/err'])[:-1]

      count = 0
      with FileReadBackwards(run+'/out') as file:
        for line in file:
          count += 1
          if 'ADJUST' in line:
            out_end = line.split('TIME =')[1][:5]
            break
          if count >= 10000:
            out_end = 'NaN'
            break

    except:
      err = np.nan
      mod_time = 0.0
      err_end = 'VOID'
      out_end = 'VOID'

    pad1 = ' ' * (26 - len(sim))
    pad2 = ' ' * (8 - len(out_end))

    if err_end == 'VOID':
      colour = u'\u001b[38;5;8m'
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

    if display or err_end != 'DONE':
      print('%s%s: %s Last time: %s %s last err: %s%s' % \
            (colour, sim, pad1, out_end, pad2, err_end.split('\n')[-1], u'\033[0m'))
