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

# Design GC number density cmap:
cdict = {'red': ((0.0, 1.000, 1.000),
                 (1.0, 1.000, 1.000)),
        'green':((0.0, 0.388, 0.388),
                 (1.0, 1.000, 1.000)),
        'blue': ((0.0, 0.278, 0.278),
                 (1.0, 0.588, 0.588))}
GC_cmap = mpl.colors.LinearSegmentedColormap('my_colormap', cdict, 256)

def plot_Nbody6(sim, sim_name, box=None, save_plot=True):

  # Initialise figure:
  fs = 10
  fig, axes = plt.subplots(figsize=(3*2, 3*2), ncols=2, nrows=2, sharex='col', sharey='row', gridspec_kw={'hspace':0.05, 'wspace':0.05})

  # Prepare plot:
  if box is None:
    box = np.abs(sim['pos'][sim['body']]).max() * 4 # pc
  panels = [(0,2), (0,1), (2,1)]
  zs = [1, 2, 0]
  labels = [('x', 'z'), ('x', 'y'), ('z', 'y')]
  point_size = (np.log10(sim['mass']) - np.log10(np.min(sim['mass'][sim['mass'] > 0])) + 0.1) * 5
  hlr = func.R_half(sim)

  # Loop over 3 spatial panels:
  for ax, panel, label, z in zip(np.ravel(axes)[[0,2,3]], panels, labels, zs):

    x, y, Z, ps = func.relief(sim['pos'][sim['body']], point_size[sim['body']], box)
    ax.scatter(sim['pos'][sim['tail'],panel[0]], sim['pos'][sim['tail'],panel[1]], s=point_size[sim['tail']], color='cornflowerblue', rasterized=True)
    ax.scatter(x, y, s=ps, c=Z, cmap=GC_cmap, rasterized=True)

    ax.set_ylim(-box/2., box/2.)
    ax.set_xlim(-box/2., box/2.)

    ax.set_xlabel('%s [kpc]' % label[0], fontsize=fs)
    ax.set_ylabel('%s [kpc]' % label[1], fontsize=fs)

    ax.label_outer()
    ax.tick_params(axis='both', labelsize=fs-2)

    ax.axvline(0, c='k', ls='--', lw=1)
    ax.axhline(0, c='k', ls='--', lw=1)

    # Add ring for the hlr:
    circle = patches.Circle((0, 0), radius=hlr, fill=False, color='k', lw=1)
    ax.add_patch(circle)

  axes[0,1].remove()

  axes[1,1].set_title(sim_name, fontsize=fs-2)

  if save_plot:
    file = '../images/raw_%s.pdf' % sim_name
    plt.savefig(file, bbox_inches='tight')
    print('Plot saved to %s.' % file)

  return
