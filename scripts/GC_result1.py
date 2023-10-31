from config import *
import numpy as np
from read_out3 import read_nbody6
import GC_functions as func

import default_setup
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
plt.ion()

from glob import glob
import pickle
import time

from scipy.ndimage import gaussian_filter

# Load the property dictionary:
#------------------------------------------------------------
with open('./files/GC_data_enhanced_mass_suite.pk1', 'rb') as f:
  GC_data = pickle.load(f)
#------------------------------------------------------------

# Custom colourmap that highlights survivors:
viridis = plt.cm.viridis_r(np.linspace(0,1,20))
loc = np.linspace(0, 1 - 0.2/0.7, 20)
cdict = {'red': tuple([(loc[i], viridis[i,0], viridis[i,0]) for i in range(20)] + [(1, 235/256, 235/256)]),
         'green': tuple([(loc[i], viridis[i,1], viridis[i,1]) for i in range(20)] + [(1, 78/256, 78/256)]),
         'blue': tuple([(loc[i], viridis[i,2], viridis[i,2]) for i in range(20)] + [(1, 54/256, 54/256)])}
cmap = mpl.colors.LinearSegmentedColormap('my_colormap', cdict, 256)

def normalise(a, normmin, normmax, vmin, vmax):
  normed = (a-vmin) / (vmax-vmin) * (normmax-normmin) + normmin
  normed[normed > normmax] = normmax
  normed[normed < normmin] = normmin
  return normed

# Create a figure instance:
fs = 12
fig, ax = plt.subplots(figsize=(6,6), ncols=1, nrows=1, gridspec_kw={'wspace':0.5})

# Loop over each simulation and add it to the plot:
for sim in list(GC_data.keys()):

  linewidths = 1.
  linecolours = cmap(normalise(GC_data[sim]['t'], 0,1, 0,13.8))
  line_segments = LineCollection([np.column_stack([[GC_data[sim]['mass'][i], GC_data[sim]['mass'][i+1]], \
                                                   [GC_data[sim]['hlr'][i], GC_data[sim]['hlr'][i+1]]]) \
                                                   for i in range(len(GC_data[sim]['t'])-1)], \
                                 linewidths=linewidths, capstyle='round', color=linecolours, rasterized=False, joinstyle='round')
  ax.add_collection(line_segments)
  if len(GC_data[sim]['t']):
    plt.plot(GC_data[sim]['mass'][-1], GC_data[sim]['hlr'][-1], 'kx', markersize=6)

ax.set_xlim(100,18000)
ax.set_ylim(0, 25)
ax.set_xscale('log')

ax.tick_params(axis='both', which='both', labelsize=fs-2)

ax.set_xlabel(r'Total bound mass [M$_{\odot}$]', fontsize=fs)
ax.set_ylabel(r'Half-light radius', fontsize=fs)

# Colorbar:
norm = mpl.colors.Normalize(vmin=0, vmax=13.8)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
ax.set_aspect('auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad='5%')
cbar = plt.colorbar(sm, cax=cax)
cbar.set_label(r'Final time + birth time [Gyr]', fontsize=fs)
cbar.ax.tick_params(labelsize=fs-2)

# Plot a histogram of end times:
fs = 12
fig, ax = plt.subplots(figsize=(6,6), ncols=1, nrows=1, gridspec_kw={'wspace':0.5})

end_times = np.empty(len(list(GC_data.keys())))
for i, sim in enumerate(list(GC_data.keys())):
  if len(GC_data[sim]['t']):
    end_times[i] = GC_data[sim]['t'][-1]
  else:
    end_times[i] = 0.
  end_times[i] = min(end_times[i], 13.8)

time_bins = np.linspace(0, 14, 15)
ax.hist(end_times, bins=time_bins, rwidth=0.9)

ax.tick_params(axis='both', which='both', labelsize=fs-2)

ax.set_xlim(0, 13.8)

ax.set_xlabel('Final time + birth time [Gyr]', fontsize=fs)
ax.set_ylabel(r'$N$', fontsize=fs)
