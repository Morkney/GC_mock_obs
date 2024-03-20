import numpy as np
import h5py
import itertools
import glob
import sys
import gc
from itertools import cycle

import stream_modules as stream

import default_setup
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as path_effects
plt.ion()

import pickle
from scipy.stats import binned_statistic_2d
from scipy.ndimage import median_filter

# Load GC data:
#--------------------------------------------------------------------
suite = 'enhanced_mass_suite'
with open('./files/GC_data_%s.pk1' % suite, 'rb') as f:
  GC_data = pickle.load(f)
suite = 'CHIMERA_suite'
with open('./files/GC_data_%s.pk1' % suite, 'rb') as f:
  GC_data2 = pickle.load(f)
suite = 'CHIMERA_massive_suite'
with open('./files/GC_data_%s.pk1' % suite, 'rb') as f:
  GC_data3 = pickle.load(f)
GC_data.update(GC_data2)
GC_data.update(GC_data3)
#--------------------------------------------------------------------

# Make a stream plot:
#--------------------------------------------------------------------
plot_type = 'mass'
stat_type = 'mean'
if (plot_type=='mass') & (stat_type=='mean'):
  c_min = 2.5
  c_max = 4.25
  clabel = r'$\log_{10}$ Mean mass [M$_{\odot}$]'
  cmap = cm.rainbow
elif (plot_type=='t') & (stat_type=='mean'):
  c_min = 0.5
  c_max = 10.25
  clabel = r'Mean time [Gyr]'
  cmap = cm.rainbow
elif (plot_type=='mass') & (stat_type=='STD'):
  c_min = 0
  c_max = 3
  clabel = r'STD Mean time [Gyr]'
  cmap = cm.inferno
elif stat_type=='STD':
  c_min = 0
  c_max = 0.25
  clabel = r'Standard deviation'
  cmap = cm.inferno
cnorm = mpl.colors.Normalize(vmin=c_min, vmax=c_max)

# Retrieve GC data:
t = np.concatenate([GC_data[i]['t'] for i in list(GC_data.keys())])
filt = t <= 13.82
c = np.concatenate([GC_data[i][plot_type] for i in list(GC_data.keys())])[filt]
if plot_type=='mass':
  c = np.log10(c)

x = [np.log10(GC_data[i]['hlr']) for i in list(GC_data.keys())]
x = np.array([median_filter(i, 20) for i in x])
vx = np.concatenate([np.append(np.diff(i), i[-1]-i[-2]) for i in x])[filt]
x = np.concatenate(x)[filt]

y = [GC_data[i]['mV'] for i in list(GC_data.keys())]
y = np.array([median_filter(i, 20) for i in y])
vy = np.concatenate([np.append(np.diff(i), i[-1]-i[-2]) for i in y])[filt]
y = np.concatenate(y)[filt]

# Parameter limits for the gradient arrays:
x_range = np.log10([10**0.1, 10**1.5])
y_range = [-10, 0.5]

# Calculate gradient arrays:
X,Y, vx,vy, density, colour = stream.streams(x, y, vx, vy, c, x_range, y_range, N_bins=31, sigma=1, type='mean', c_type=stat_type)
density -= 0.5
density[density < 0] = 0

fs = 12
fig, ax = plt.subplots(figsize=(6,6))
#ax.streamplot(X,Y, vx,vy, density=2.25, color=colour, linewidth=density, norm=cnorm, cmap=cmap, minlength=0.2)
ax.streamplot(X,Y, vx,vy, density=2.2, color=colour, linewidth=density, norm=cnorm, cmap=cmap, minlength=0.2)

ax.tick_params(axis='both', labelsize=fs-2)

ax.set_xlim(0.1, 1.5)
ax.set_ylim(0, -8)

# Change x-labels to logspace:
log_ax = ax.twiny()
log_ax.set_xscale('log')
log_ax.set_xlim(10**0.1, 10**1.5)
from matplotlib.ticker import FormatStrFormatter
log_ax.xaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
log_ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
ax.set_xticks([])
log_ax.xaxis.set_ticks_position('bottom')
log_ax.xaxis.set_label_position('bottom')
log_ax.tick_params(which='both', labelsize=fs-2, top=True, bottom=True)

log_ax.set_xlabel('Half-light radius [pc]', fontsize=fs)
ax.set_ylabel('Absolute V-band magnitude', fontsize=fs)

# Add resolution limit:
ax.axvspan(0, np.log10(3), fc='k', alpha=0.2, zorder=99)
#--------------------------------------------------------------------

# Add colourbar:
#--------------------------------------------------------------------
cbar_width = 0.05
cbar_pad = 0.025

norm = mpl.colors.Normalize(vmin=c_min, vmax=c_max)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
divider = make_axes_locatable(ax)
l, b, w, h = ax.get_position().bounds
cax = fig.add_axes([l+w+cbar_pad, b, w*cbar_width, h])
cbar = plt.colorbar(sm, cax=cax)
cbar.set_label(clabel, fontsize=fs)
cbar.ax.tick_params(labelsize=fs-2)
#--------------------------------------------------------------------

plt.savefig('../images/streamplot_%s_%s.pdf' % (plot_type, stat_type), bbox_inches='tight')
