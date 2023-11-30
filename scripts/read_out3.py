#!/usr/bin/python
import numpy as np
from scipy.io import FortranFile
import os, re

values = None
def extract(N):
  global values
  res = values[:N]
  values = values[N:]
  return res

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def read_nbody6(file, df=False):

  global values

  run_num = 0
  runs = []

  # Find all runs:
  for root, dirs, files in os.walk(file):
    if 'OUT3' in files:
      runs.append(os.path.join(root, 'OUT3'))
  runs = sorted(runs, key=natural_key)

  if any('run1' in x for x in runs):
    input = open(file + '/run1/fort.10')
    if 'run1' not in runs[0]:
      runs = runs[1:] + [runs[0]]
  else:
    input = open(file + '/fort.10')

  # Grab initial velocity for normalisation purposes:
  lines = input.readlines()
  result = np.array([])
  for x in lines:
    result = np.append(result, float(x.split()[4]))
  velocity = result.max()

  # Grab initial velocity again:
  input = open(file + '/GC_IC.input')
  input = input.read().split('\n')
  try:
    velocity2 = float(input[12].split(' ')[3])
  except:
    print('Assuming this is a DMC sim.')
    velocity2 = float(input[13].split(' ')[3])

  f_in = FortranFile(runs[run_num])

  simulation = []
  rmax = np.inf

  first = True
  hollow_counter = 0
  while True:

    try:
      header = f_in.read_ints()
      ntot, model, nrun, nk = header
    except:
      run_num += 1
      if run_num >= len(runs):
        f_in.close()
        order = np.array([simulation[i]['age'] for i in range(len(simulation))])
        simulation = np.array(simulation)[order.argsort()]
        return simulation
      else:
        f_in.close()
        f_in = FortranFile(runs[run_num])
        print('Read all OUT3 files.')
        continue

    try:
      values = f_in.read_reals(dtype=np.float32)
    except:
      print('No values.')
      continue

    # Read in the body values:
    alist = extract(nk)

    bodys = extract(ntot)
    xs = extract(ntot*3).reshape((ntot, 3))
    vs = extract(ntot*3).reshape((ntot, 3))
    ids = extract(ntot)
    bounds = extract(ntot)
    kstara = extract(ntot)
    lum = extract(ntot)
    teff = extract(ntot)

    # Define velocity correction:
    if first:
      vel_correct = velocity / vs[:,0].max()
      vel_correct2 = velocity2 / alist[12]

    # Define outputs:
    snapshot = {}
    snapshot['mass'] = bodys * alist[3]
    snapshot['pos'] = xs * alist[2]
    snapshot['vel'] = vs * vel_correct
    snapshot['ids'] = (ids / 1e-45).astype('int')
    snapshot['nbound'] = (bounds / 1e-45).astype('bool')
    snapshot['kstara'] = (kstara / 1e-45).astype('int')
    snapshot['lum'] = 10**lum
    snapshot['teff'] = teff

    snapshot['age'] = alist[0] * alist[4]
    snapshot['nbin'] = int(alist[1])
    snapshot['rdens'] = np.array(alist[6:9] * alist[2])
    snapshot['rg'] = np.array(alist[9:12])
    snapshot['vg'] = np.array(alist[12:15]) * vel_correct2
    snapshot['rcore'] = np.array(alist[2] * alist[15])

    if df:
      snapshot['R_tide'] = alist[20] * alist[2]
      snapshot['vcore'] = np.array(alist[24:27]) * vel_correct2
      snapshot['coulog'] = alist[21]
      snapshot['nroche'] = alist[22]
      snapshot['dynfcoef'] = alist[23]
      snapshot['rhmroche'] = alist[27] * alist[2]
      snapshot['ncore'] = int(alist[16])
    else:
      snapshot['R_tide'] = alist[20]

    snapshot['alist'] = alist

    # Remove outliers:
    r = np.linalg.norm(snapshot['pos'][snapshot['nbound']] - snapshot['rdens'], axis=1)
    percent = max(20, int(snapshot['nbound'].sum() / 100))
    outliers = r >= np.median(np.sort(r)[-percent:]) + 1.
    snapshot['nbound'][np.where(snapshot['nbound']) and np.where(outliers)] = False

    # Do not append if there are less than ten bound particles:
    if snapshot['nbound'].sum() < 10:
      print('>    Less than 10 bound particles.')
      break

    # Kill condition for hollow clusters:
    if ((r < 10).sum() < percent) & (np.linalg.norm(snapshot['rg']*1e3 - snapshot['rdens']) < 10):
      print('>    Hollow cluster.')
      hollow_counter += 1
      if hollow_counter > 10: # EDITED!
        break
      else:
        continue

    # Carry forward smallest Roche region [ignore sudden drops]:
    '''
    if first or (max(r)*1.5 > rmax):
      rmax = min(rmax, max(r)*1.01)
      r_full = np.linalg.norm(snapshot['pos'] - snapshot['rdens'], axis=1)
      snapshot['nbound'][r_full > rmax] = False

      # Kill condition for hollow clusters:
      if ((r_full < 10).sum() < percent) & (np.linalg.norm(snapshot['rg']*1e3 - snapshot['rdens']) < 10):
        print('>    Hollow cluster.')
        hollow_counter += 1
        if hollow_counter > 10: # EDITED!
          break
        else:
          continue
    '''

    hollow_counter = 0

    simulation.append(snapshot)
    first = False

    #if snapshot['age'] > 200: break  ####

  return simulation
