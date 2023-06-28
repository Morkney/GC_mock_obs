from config import *

import numpy as np
from read_out3 import read_nbody6
import GC_functions as func

from scipy.ndimage import gaussian_filter
from scipy.interpolate import interpn

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt, matplotlib.patches as patches
plt.ion()

print('Converting Nbody6 GC data to format required by COCOA tools.')

# Load the simulation data:
#--------------------------------------------------------------------------
sim = read_nbody6(path+sim_name, df=True)
print('>    %s has been loaded.' % sim_name)

# Select the relevant snapshot:
snapshot = -1
s = sim[snapshot]
time = s['age']
print('>    Snapshot %i with time %.2f Myr' % (snapshot, time))
#--------------------------------------------------------------------------

# The needed data are the x/y positions and the mass:
#--------------------------------------------------------------------------
# Find tail stars, body stars, and black hole type stars:
tail = ~s['nbound'] & (s['mass'] > 0)
body = s['nbound'] & (s['mass'] > 0)
BHs = s['kstara'] == (13 and 14)

# Centre on CoM:
mass = s['mass']
pos = s['pos'] - s['rdens']
cen = np.average(pos[body & ~BHs], axis=0, weights=s['mass'][body & ~BHs])
pos -= cen
s['r'] = np.linalg.norm(pos, axis=1)
print('>    Centred on the GC body.')

# Rotate the stars so that the galaxy points right:
pos_G = -(s['rg'] + s['rdens'] / 1e3 + cen/1e3) * 1e3
R_G = np.linalg.norm(pos_G)
func.alignment(s, cen)
print('>    Aligned such that the galactic centre is to the right.')
#--------------------------------------------------------------------------

# Save data to new file:
#--------------------------------------------------------------------------
s['z'] = np.ones_like(s['mass']) * 0.001

# Data for input into YBC:
file = path + sim_name + '/reduced_data_%s.dat' % sim_name
data = np.transpose([*np.transpose(pos), np.log10(s['lum']), s['teff'], s['mass'], s['z'], s['kstara']])
format = '%.3f %.3f %.3f %.3e %.3f %.3f %.3e %i'
np.savetxt(file, data, header='x/pc, y/pc, z/pc, log(Lsol), Teff/K, m/Msol, Z, Phase', fmt=format)
print('>    Reduced data saved to %s.' % file)

# YBC parameters:
# 1. JWST MIRI wide filters, Vegamags
# 2. a) Default option
# 2. b) Default option
# 2. c) Default option
# 3. 0 (No extinction)
# 4. A file containing stars.
#    logL: 4
#    Teff(K): 5
#    mass: 6
#    Z: 7
#    Mdot: ?
#--------------------------------------------------------------------------

# Make a plot to confirm that the data is sensible:
#--------------------------------------------------------------------------
fs = 10
fig, axes = plt.subplots(figsize=(3*2, 3*2), ncols=2, nrows=2, sharex='col', sharey='row', gridspec_kw={'hspace':0.05, 'wspace':0.05})

box = np.abs(pos[body]).max() * 4 # pc
panels = [(0,2), (0,1), (2,1)]
zs = [1, 2, 0]
labels = [('x', 'z'), ('x', 'y'), ('z', 'y')]
point_size = (np.log10(mass) - np.log10(np.min(mass[mass > 0])) + 0.1) * 5
hlr = func.R_half(s)

# Orbital trajectory [Not included on the plot, not sure if I have the units right etc.]:
vel = s['vel']
vcen = np.average(vel[body & ~BHs], axis=0, weights=s['mass'][body & ~BHs])
vel_G = s['vg'] - vcen

# Design GC number density cmap:
cdict = {'red': ((0.0, 1.000, 1.000),
                 (1.0, 1.000, 1.000)),
        'green':((0.0, 0.388, 0.388),
                 (1.0, 1.000, 1.000)),
        'blue': ((0.0, 0.278, 0.278),
                 (1.0, 0.588, 0.588))}
GC_cmap = mpl.colors.LinearSegmentedColormap('my_colormap', cdict, 256)

for ax, panel, label, z in zip(np.ravel(axes)[[0,2,3]], panels, labels, zs):

  x, y, Z, ps = func.relief(pos[body], point_size[body], box)
  ax.scatter(pos[tail,panel[0]], pos[tail,panel[1]], s=point_size[tail], color='cornflowerblue', rasterized=True)
  ax.scatter(x, y, s=ps, c=Z, cmap=GC_cmap, rasterized=True)

  ax.set_ylim(-box/2., box/2.)
  ax.set_xlim(-box/2., box/2.)

  ax.set_xlabel('%s [kpc]' % label[0], fontsize=fs)
  ax.set_ylabel('%s [kpc]' % label[1], fontsize=fs)

  ax.label_outer()
  ax.tick_params(axis='both', labelsize=fs-2)

  ax.axvline(0, c='k', ls='--', lw=1)
  ax.axhline(0, c='k', ls='--', lw=1)

  circle = patches.Circle((0, 0), radius=hlr, fill=False, color='k', lw=1)
  ax.add_patch(circle)

axes[0,1].remove()

axes[1,1].set_title(sim_name, fontsize=fs+2)

plt.savefig('./images/raw_%s.pdf' % sim_name, bbox_inches='tight')
#--------------------------------------------------------------------------
