from config import *
import numpy as np
from read_out3 import read_nbody6
import GC_functions as func
import sys

import default_setup
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

# Load chosen simulation:
#------------------------------------------------------------
#sim_name = 'Halo605_fiducial_hires_output_00036_40'
sim_name = sys.argv[1]
s = read_nbody6(path + f'Nbody6_sims/{suite}/{sim_name}', df=True)
#------------------------------------------------------------

# Orbit, mass, size:
fs = 12
fig, ax = plt.subplots(figsize=(9,3), ncols=3, nrows=1, gridspec_kw={'wspace':0.5})

time = np.empty(len(s))
orbit_pos = np.empty([len(s), 3])
mass = np.empty(len(s))
size = np.empty(len(s))
for i, s_i in enumerate(s):

  time[i] = s_i['age']/1e3

  # Centre the GC position:
  body_noBHs = s_i['nbound'] & (s_i['kstara'] != 14)
  cen = np.average(s_i['pos'][body_noBHs], weights=s_i['mass'][body_noBHs], axis=0)

  # Orbital position:
  orbit_pos[i] = s_i['rdens'] + s_i['rg']*1e3 #+ cen

  mass[i] = s_i['mass'][s_i['nbound']].sum()

  s_i['pos'] -= cen
  s_i['r'] = np.linalg.norm(s_i['pos'], axis=1)
  size[i] = func.R_half(s_i)

# Plot the orbital history:
ax[0].plot(time, np.abs(orbit_pos), lw=0.5, label=[r'$x$',r'$y$',r'$z$'])
ax[0].plot(time, np.linalg.norm(orbit_pos, axis=1), 'k-', label=r'$R$')
#ax[0].legend(fontsize=fs-2)

ax[0].set_xlabel(r'Time [Gyr]', fontsize=fs)
ax[0].set_ylabel(r'Radius [pc]', fontsize=fs)

# Plot the mass history:
ax[1].plot(time, mass, 'k-')
ax[1].set_xlabel(r'Time [Gyr]', fontsize=fs)
ax[1].set_ylabel(r'Mass [M$_{\odot}$]', fontsize=fs)

# Plot the size history:
ax[2].plot(time, size, 'k-')
ax[2].set_xlabel(r'Time [Gyr]', fontsize=fs)
ax[2].set_ylabel(r'Half-light radius [pc]', fontsize=fs)

def square(ax):
  ax.set_aspect(np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()))
  return

for i in range(3):
  ax[i].set_aspect('auto')
  ax[i].tick_params(which='both', axis='both', labelsize=fs-2)
  square(ax[i])

string = suite + '\n' + sim_name
fig.suptitle(string, fontsize=fs)
