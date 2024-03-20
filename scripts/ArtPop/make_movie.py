from config import *
import numpy as np
from read_out3 import read_nbody6

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib.collections import LineCollection
#plt.ion()

from scipy.interpolate import interpn, BSpline, make_interp_spline
from scipy.ndimage import gaussian_filter

import time
import os
import subprocess
import sys

import multiprocessing
import imageio

# Load chosen simulation:
#------------------------------------------------------------
sim_name = 'DM_test_new'
s = read_nbody6(path + f'Nbody6_sims/{suite}/{sim_name}', df=True)
#------------------------------------------------------------

# Various setups:
#------------------------------------------------------------
fs = 10

# Design GC colourmap:
CM = cm.get_cmap('RdPu')
CM._init()
RGB_max = 255.
CM._lut[:-3,0] = 255 / RGB_max
CM._lut[:-3,1] = np.linspace(99, 255, CM.N) / RGB_max
CM._lut[:-3,2] = np.linspace(71, 150, CM.N) / RGB_max

# Window sizes:
GC_box_width = 20.
GC_box_pad = 20.
follow = True
box_width = abs(np.array([i['rg'] for i in s])).max() * 1e3 + GC_box_width + GC_box_pad

# Make temporary frame storage:
if not os.path.isdir('./pngs'):
  os.mkdir('./pngs')
else:
  os.system('rm -rf ./pngs')

# Initialise arrays:
trace = np.array([])
trace_fade = int(5e1 / s[1]['age'])
times = np.array([i['age'] for i in s])

def R_z(theta):
  return np.array([[np.cos(theta),-np.sin(theta),0.], \
                   [np.sin(theta),np.cos(theta),0.], \
                   [0.,0.,1.]])

def angle(a, b):
  return np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
#------------------------------------------------------------

# Function to plot a single Nbody6 simulation snapshot frame:
#------------------------------------------------------------
def frame(i):

  global s

  print('>    Snapshot %i ' % i)
  if s[i]['nbound'].sum() == 0:
    return

  fig, ax = plt.subplots(figsize=(8,4), ncols=2, nrows=1, gridspec_kw={'wspace':0.3})

  # SC-centric and G-centric coordinates:
  pos_SC = s[i]['pos'] - s[i]['rdens']
  pos_G = s[i]['pos'] + s[i]['rg']*1e3

  # Turn stellar mass into particle size:
  mass_size = (np.log10(s[i]['mass'][s[i]['mass'] > 0]) - \
               np.log10(s[i]['mass'][s[i]['mass'] > 0]).min() + 0.1) * (1/3.)

  # Set up number density colourmap:
  count, x_e, y_e = np.histogram2d(pos_SC[s[i]['nbound'],0], pos_SC[s[i]['nbound'],1], bins=51)
  z = interpn(((x_e[1:] + x_e[:-1])/2., (y_e[1:]+y_e[:-1])/2.), count, \
              np.vstack([pos_SC[:,0],pos_SC[:,1]]).T, method="linear", bounds_error=False)
  idx = z.argsort()
  pos_SC, pos_G, z, mass_size = pos_SC[idx], pos_G[idx], np.log10(z[idx]), mass_size[idx]
  z[z != z] = np.nanmin(z)

  # Track the orbit:
  trace = np.vstack([s[j]['rdens'] + s[j]['rg']*1e3 for j in range(np.max([0, 1+i-trace_fade]), i+1)])
  k = max(0, min(3, len(trace)-1))
  orig_res = np.linspace(0, 1, len(trace))
  spline_res = np.linspace(0, 1, len(trace) * 5)
  trace = make_interp_spline(orig_res, trace, k=k)(spline_res)

   # Get GC velocity vector and rotation:
  if follow:
    if i > 0:
      vec = np.array([trace[-1][0] - trace[-2][0], trace[-1][1] - trace[-2][1]])
      theta_xy = angle(vec, [1.,0.]) * np.sign(-vec[1])
    else:
      vec = [1, 1, 1]
      theta_xy = 0
    rotate = R_z(theta_xy)

    # Rotations:
    pos_SC = np.dot(rotate, pos_SC.T).T

  # Build the plot:
  #-----------------------------------------------------------------------------
  # Split into tail stars and body stars:
  s[i]['body'] = (s[i]['nbound'] & (s[i]['mass'] > 0))[idx]
  s[i]['tail'] = (~s[i]['nbound'] & (s[i]['mass'] > 0))[idx]

  ax[0].scatter(pos_G[s[i]['tail'],0], pos_G[s[i]['tail'],1],s=mass_size[s[i]['tail']], color='cornflowerblue')

  line_colour = (np.ones([trace_fade*5, 3]) * np.vstack(np.arange(trace_fade*5)[::-1]))[-len(trace):] / (trace_fade*5)
  trace_line = LineCollection([np.column_stack([[trace[i,0], trace[i+1,0]], [trace[i,1], trace[i+1,1]]]) for i in range(len(trace)-1)], \
                              linewidths=1, capstyle='round', color=line_colour, joinstyle='round', zorder=0)
  ax[0].add_collection(trace_line)

  ax[0].scatter(pos_G[s[i]['body'],0], pos_G[s[i]['body'],1], s=mass_size[s[i]['body']], c=z[s[i]['body']], cmap=cm.RdPu)
  square = patches.Rectangle((trace[-1][0] - GC_box_width, trace[-1][1] - GC_box_width), \
                             width=GC_box_width*2, height=GC_box_width*2, \
                             facecolor='None', edgecolor='k', lw=1, ls='--')

  if follow:
    square_rotate = mpl.transforms.Affine2D().rotate_deg_around(\
                    trace[-1][0], trace[-1][1], -theta_xy * 180./np.pi) \
                    + ax[0].transData
    square.set_transform(square_rotate)

  ax[0].add_patch(square)
  ax[1].scatter(pos_SC[s[i]['tail'],0], pos_SC[s[i]['tail'],1], alpha=1, s=mass_size[s[i]['tail']] * 2., color='cornflowerblue')
  ax[1].scatter(pos_SC[s[i]['body'],0], pos_SC[s[i]['body'],1], alpha=1, s=mass_size[s[i]['body']] * 2., c=z[s[i]['body']], cmap=cm.RdPu)

  ax[0].plot([0,0], [0,0], 'k*', markersize=10)
  ax[0].text(0.95, 0.95, r'Age$ = %.2f\,$Gyr' % (s[i]['age'] / 1e3), va='top', ha='right', transform=ax[0].transAxes)
  ax[0].text(0.05, 0.95, r'%i' % i, va='top', ha='left', transform=ax[0].transAxes)

  # Manage the axes:
  for axis, width in zip([0,1], [box_width, GC_box_width]):
    ax[axis].set_ylim(-width, width)
    ax[axis].set_xlim(-width, width)

    ax[axis].set_yticks(ax[axis].get_xticks())
    ax[axis].set_xticks(ax[axis].get_xticks())

    ax[axis].set_xlabel('X [pc]', fontsize=fs)
    ax[axis].set_ylabel('Y [pc]', fontsize=fs)

    ax[axis].tick_params(axis='both', labelsize=fs-2)

    ax[axis].set_aspect(1)

  plt.savefig('./pngs/movie_frame_%05d.png' % i, output='pdf', bbox_inches='tight')

  plt.close(fig)

  return
#------------------------------------------------------------

frames = []
n_threads = 8
pool = multiprocessing.Pool(n_threads)
tasks = np.arange(len(s))
pool.map(frame, tasks)

frames = np.array([imageio.imread('./pngs/movie_frame_%05d.png' % i) for i in range(0, len(s), 1)])
writer = imageio.get_writer('../images/Movie_%s.mp4' % sim_name, fps=3)
for frame in frames:
  writer.append_data(frame)
writer.close()

# Remove temporary frame storage:
os.system('rm -rf ./pngs')

print('Done.')

