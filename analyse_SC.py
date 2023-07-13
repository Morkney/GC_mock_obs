from config import *

import numpy as np
from read_out3 import read_nbody6
import GC_functions as func
import plot_Nbody6_data

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt, matplotlib.patches as patches
plt.ion()

fig, ax = plt.subplots(figsize=(8,4), ncols=2, nrows=1, gridspec_kw={'wspace':0.3})

for sim_name, label in zip(['DM_test', 'DM_test_new'], ['Default', 'With DMC']):

  # Load the simulation data:
  #--------------------------------------------------------------------------
  sim = read_nbody6(path+sim_name, df=True)
  print('>    Nbody6 simulation %s has been loaded.' % sim_name)

  time = np.array([s['age'] for s in sim])
  #--------------------------------------------------------------------------

  # Track the full orbit:
  #--------------------------------------------------------------------------
  orbit_pos = np.empty([len(sim), 3])
  hmr = np.empty(len(sim))
  for i, s_i in enumerate(sim):

    # Centre the GC position:
    body_noBHs = s_i['nbound'] & (s_i['kstara'] != 14)
    cen = np.average(s_i['pos'][body_noBHs], weights=s_i['mass'][body_noBHs], axis=0)

    # Orbital position:
    #orbit_pos[i] = s_i['rdens'] + s_i['rg']*1e3 + cen
    #orbit_pos[i] = s_i['rg']*1e3 + cen
    orbit_pos[i] = s_i['rg']*1e3 + s_i['rdens']

    # Calculate half-mass:
    #s_i['pos'] -= cen
    s_i['r'] = np.linalg.norm(s_i['pos'] - cen, axis=1)
    hmr[i] = func.R_half(s_i)

    #plt.plot(s_i['pos'][:,0] + orbit_pos[i][0], s_i['pos'][:,1] + orbit_pos[i][1], 'k,', alpha=0.1)
  #--------------------------------------------------------------------------

  # Plot the result:
  #--------------------------------------------------------------------------
  ax[0].plot(time, hmr, label=label)
  ax[1].plot(time, np.linalg.norm(orbit_pos, axis=1), label=label)
  #--------------------------------------------------------------------------

ax[0].set_xlabel('Time [Myr]')
ax[0].set_ylabel('Half mass radius [pc]')
ax[1].set_xlabel('Time [Myr]')
ax[1].set_ylabel('Orbital radius [pc]')
ax[1].legend()

#plt.close()
sim2 = read_nbody6(path+'DM_test', df=True)

def pp(s):
  #plt.plot(s['pos'][:,0] + s['rg'][0]*1e3 - s['rdens'][0], s['pos'][:,1] + s['rg'][1]*1e3 - s['rdens'][1], 'k,')
  body = s['nbound']
  plt.plot(s['pos'][body,0] + s['rg'][0]*1e3, s['pos'][body,1] + s['rg'][1]*1e3, 'k,')
  plt.plot(s['pos'][~body,0] + s['rg'][0]*1e3, s['pos'][~body,1] + s['rg'][1]*1e3, 'b,')
  plt.plot(s['rdens'][0] + s['rg'][0]*1e3, s['rdens'][1] + s['rg'][1]*1e3, 'rx')
  plt.xlim(-150,150)
  plt.ylim(-150,150)
  plt.axvline(0, c='grey', lw=1)
  plt.axhline(0, c='grey', lw=1)
  plt.gca().set_aspect(1)
  ring = patches.Circle((0,0), 100, facecolor='None', edgecolor='r', linewidth=0.5, zorder=100)
  plt.gca().add_patch(ring)

print(1/0)
plt.subplots()
pp(sim[60])
plt.subplots()
pp(sim2[59])
