# Personal modules:
from config import *
from read_out3 import read_nbody6
import GC_functions as func
import interpolation
import ArtPop_movie_frame as mf

# Maths modules:
import numpy as np
from scipy.ndimage import zoom

# Standard modules:
import time
import os
import pickle
import io

# Imaging modules:
import multiprocessing
import imageio
from PIL import Image

# Astro modules:
import artpop
from astropy import units as u
from astropy.visualization import make_lupton_rgb

# Pyplot modules:
import default_setup
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()

# Science modules:
from scipy.interpolate import make_interp_spline

import agama
from scipy.interpolate import make_interp_spline, CubicHermiteSpline
agama.setUnits(mass=1, length=1e-3, velocity=1)
timeunit = 0.97779222

def sigmoid(x, beta=3.):
  return 1 / (1 + (x / (1-x))**(-beta))

# REMAINING PROBLEMS:
# There is a discontinuity joining the 'interp' and 'standard' simulation instances. Fix it.
# The trace is too long for the interpolated snapshots. Fix it.
# Now the trace is missing for the interpolated snapshots. Fix it.

# Load chosen simulation:
#------------------------------------------------------------
suite = 'enhanced_mass_suite'
sim = 'Halo1459_fiducial_hires_output_00023_3'
s = read_nbody6(path+suite+'/'+sim, df=True)

# Load some simulation parameters:
data = np.genfromtxt('./files/GC_property_table.txt', unpack=True, skip_header=2, dtype=None)
GC_ID = np.where([int(sim.split('_')[-1]) == i[15] for i in data])[0][0]
GC_Z = data[GC_ID][7]
GC_birthtime = data[GC_ID][8]
for i in range(len(s)):
  s[i]['age'] += GC_birthtime

# Sanitise parameters:
GC_Z = max(min(GC_Z, 0.5), -3.)

rng = 100
#------------------------------------------------------------

# Various setups:
#------------------------------------------------------------
# Window sizes:
with open('./files/GC_data_%s.pk1' % suite, 'rb') as f:
  GC_data = pickle.load(f)
params = {}
params['GC_width'] = 20. # [pc]
params['GC_pad'] = 10. # [pc]
params['full_width'] = GC_data[sim]['rg'].max() + params['GC_width'] + params['GC_pad']
params['follow'] = True
params['GC_Z'] = GC_Z

# Distance bar scales:
for panel in ['full', 'GC']:
  params['%s_ruler' % panel] = int('%.0f' % float('%.1g' % (params['%s_width' % panel]/2.1)))
  params['%s_corner1' % panel] = params['%s_width' % panel] - (0.1 * params['%s_width' % panel]) - params['%s_ruler' % panel]
  params['%s_corner2' % panel] = 0.9 * params['%s_width' % panel]
  params['%s_cap' % panel] = 0.025 * params['%s_width' % panel]

# Calculate rotation matrix to place the GC orbit in the x-y plane:
params['rotation'] = func.align_orbit(GC_data[sim]['posg'][0], GC_data[sim]['velg'][0])
#------------------------------------------------------------

def ff(axis, colour):
  axis.plot([1-0.1, 1-0.05], [0.1, 0.1], marker=5, lw=0, color=colour, markersize=24, transform=axis.transAxes)

def frame(task):
  i, rotation, key = task[0], task[1], task[2]

  fs = 10
  fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=1, gridspec_kw={'wspace':0.05})

  print('>    Snapshot %i ' % i)
  if s[i]['nbound'].sum() == 0:
    return

  # Add the metadata:
  #-----------------------------------------------------------------------------
  ax[0].text(0.025, 0.975, mf.metadata(s[i]), va='top', ha='left', color='w', \
             fontsize=fs-2, transform=ax[0].transAxes, path_effects=mf.paths, fontname='monospace', zorder=98)
  #-----------------------------------------------------------------------------

  # Analyse SC data:
  #-----------------------------------------------------------------------------
  # SC-centric and G-centric coordinates:
  pos_SC = np.dot(s[i]['pos'] - s[i]['shrink_centre'], params['rotation'].T)
  pos_G = np.dot(s[i]['pos'] + s[i]['rg']*1e3, params['rotation'].T)

  # Track the orbit:
  trace, theta_xy = mf.orbit_trace(s, i, ax[0], params)
  pos_SC = np.dot(pos_SC, func.R_z(theta_xy).T)

  # Cull remnants:
  s[i]['remnants'] = np.array([kstara in [10, 11, 12, 13, 14, 15] for kstara in s[i]['kstara']])
  s[i]['mass'] = s[i]['mass'][~s[i]['remnants']]
  #-----------------------------------------------------------------------------

  # Add images to the plot:
  #-----------------------------------------------------------------------------
  # Select rescaling for orbit panel: [I don't quite understand this part!]
  orbit_xy_dim, zoom_xy_dim = 1001, 501
  orbit_xy_dim_ = zoom_xy_dim/(params['GC_width']/params['full_width'])/mf.psf_fractions
  choice = np.argmin(np.abs(orbit_xy_dim_ - orbit_xy_dim))

  # Loop over each panel type:
  widths = [params['full_width'], params['GC_width']]
  stretches = [mf.img_stretches[choice], 0.2]
  fractions = [mf.psf_fractions[choice], 1]
  for j, (pos, width, stretch, fraction) in enumerate(zip([pos_G, pos_SC], widths, stretches, fractions)):
    s[i]['pos'] = pos[~s[i]['remnants']]
    zoom_amount = params['GC_width'] / width
    xy_dim = int(zoom_xy_dim/zoom_amount/fraction)
    if not xy_dim%2:
      xy_dim += 1

    # ArtPop source:
    src = artpop.MISTNbodySSP(
      log_age = np.log10(s[i]['age']*1e6),
      feh = params['GC_Z'],
      r_eff = width * u.pc,
      num_r_eff = 1,
      use_stars = s[i],
      phot_system = 'HST_ACSWF',
      distance = 30 * u.kpc,
      xy_dim = xy_dim,
      pixel_scale = 0.05 * zoom_amount * u.arcsec/u.pixel,
      random_state = rng)

    # Mock observe:
    images = []
    for b in mf.bands:
      if j:
        _psf = mf.psf[b]
      else:
        _psf = zoom(mf.psf[b], zoom=1/fraction, order=1)
        _psf *= np.shape(mf.psf[b])[0]**2 / np.shape(_psf)[0]**2
      obs = mf.imager.observe(src, b, exptime=mf.exptime[b], psf=_psf, zpt=22)
      images.append(obs.image * mf.scale[b])
    rgb = make_lupton_rgb(*images, stretch=stretch, Q=8)

    # Add this to the plot:
    ax[j].imshow(rgb, origin='lower', extent=np.array([-1,1,-1,1])*width)
  #-----------------------------------------------------------------------------

  # Add other miscellaneous information:
  #-----------------------------------------------------------------------------
  # Add a marker to indicate the centre of the galactic potential:
  ax[0].plot([0,0], [0,0], 'w*', markersize=12, markeredgewidth=1, markeredgecolor='k')

  # Panel-specific information:
  for j, (panel, label) in enumerate(zip(['full', 'GC'], ['Orbital plane', 'Zoom-in'])):
    # Distance bars:
    mf.distance_bar(ax[j], panel, params)

    # Panel labels:
    ax[j].text(0.025, 0.025, label, va='bottom', ha='left', color='w', \
               fontsize=fs, transform=ax[j].transAxes, path_effects=mf.paths)

  # Add fast-forward symbol:
  if key==1: ff(ax[0], 'w')
  #-----------------------------------------------------------------------------

  # Manage the axes:
  #-----------------------------------------------------------------------------
  for axis, width in zip([0,1], [params['full_width'], params['GC_width']]):
    ax[axis].set_aspect(1)
    ax[axis].set_yticks([]) ; ax[axis].set_xticks([])
    ax[axis].set_ylim(-width, width) ; ax[axis].set_xlim(-width, width)
  #-----------------------------------------------------------------------------

  # Save frame as a compressed png file:
  #-----------------------------------------------------------------------------
  ram = io.BytesIO()
  plt.savefig(ram, format='png', bbox_inches='tight', dpi=300)
  ram.seek(0)
  img = Image.open(ram)
  img_compressed = img.convert('RGB').convert('P', palette=Image.ADAPTIVE)
  img_compressed.save('./pngs2/movie_frame_%05d.png' % i, format='PNG')
  #plt.close(fig)
  #-----------------------------------------------------------------------------

  return

#------------------------------------------------------------

def show(n, s=s):
  pos = np.array([i['pos'][n] for i in s])
  t = np.array([i['age'] for i in s])
  plt.plot(pos[:,0], pos[:,1])
  return

end_time = min(1e3*13.82, s[-1]['age'])

# Super-high fidelity beginning:
time_span = 100 # 1530 # [Myr]
t_min = 0 + GC_birthtime # [Myr]
t_max = 0 + time_span + GC_birthtime # [Myr]
#t_min = 1400
#t_max = 1440
delta_t = 0.5 # [Myr]


# Calculate bulk properties of the star cluster:
#------------------------------------------------------------
for i in range(len(s)):
  sorted_IDs = np.argsort(s[i]['ids'])
  for field in ['mass', 'pos', 'vel', 'kstara', 'nbound']:
    s[i][field] = s[i][field][sorted_IDs]
    s[i][field] = s[i][field][:len(s[0][field])]
  s[i]['shrink_centre'] = func.shrink(s[i])
  s[i]['pos'] -= s[i]['shrink_centre']
  s[i]['r'] = np.linalg.norm(s[i]['pos'], axis=1)
  s[i]['hlr'] = func.R_half(s[i], type='mass', filt=[0., 100.])
  s[i]['vcore'] = np.average(s[i]['vel'][s[i]['r'] <= np.percentile(s[i]['r'][s[i]['nbound']], 60)], axis=0)
  s[i]['pos'] += s[i]['shrink_centre']
#------------------------------------------------------------


#===============================#
# Take a time-slice of the simulation data:
times = np.array([i['age'] for i in s])
time_slice = (times >= t_min) & (times <= t_max)
s_slice = [s[i] for i in np.where(time_slice)[0]]

# Define the new set of timesteps using delta_t:
cadence_multiplier = round((s_slice[1]['age'] - s_slice[0]['age']) / delta_t)
interpolated_steps = cadence_multiplier + 1
total_time = s_slice[1]['age'] - s_slice[0]['age']
#===============================#


#===============================#
s_interp = []
A = A = sigmoid(np.linspace(0, 1, interpolated_steps))
for i in range(len(s)):

  # Centre the star cluster:
  s[i]['pos'] -= s[i]['shrink_centre']
  s[i]['vel'] -= s[i]['vcore']

  # Make a spherically symmetric profile fit:
  rmax = s[i]['r'][s[i]['nbound']].max()
  data_snap = (np.concatenate([s[i]['pos'], s[i]['vel']], axis=1), s[i]['mass'])
  potential = agama.Potential(type='Multipole', particles=data_snap, gridSizeR=100, rmin=1e-3, rmax=rmax, symmetry='Spherical')

  # Integrate the orbits over the backward timestep and interpolate:
  if i > 0:
    orbits = agama.orbit(ic=data_snap[0], potential=potential, time=-total_time, trajsize=interpolated_steps)
    for j in range(1, interpolated_steps-1):
      pos_vel = np.vstack([k[1][j] for k in orbits])
      s_interp[-j]['pos'] = (1-A[j])*pos_vel[:,[0,1,2]] + A[j]*s_interp[-j]['pos']
      s_interp[-j]['vel'] = (1-A[j])*pos_vel[:,[3,4,5]] + A[j]*s_interp[-j]['vel']

  # Calculate orbits over the forward timestep:
  if i==len(s) - 1:
    snapshot = {}
    snapshot['age'] = s[i]['age']
    snapshot['pos'] = s[i]['pos']
    snapshot['vel'] = s[i]['vel']
    snapshot['shrink_centre'] = s[i]['shrink_centre']
    s_interp.append(snapshot)
  else:
    orbits = agama.orbit(ic=data_snap[0], potential=potential, time=total_time, trajsize=interpolated_steps)
    for j in range(interpolated_steps-1):
      snapshot = {}
      snapshot['age'] = s[i]['age'] + orbits[0][0][j]
      pos_vel = np.vstack([k[1][j] for k in orbits])
      snapshot['pos'] = pos_vel[:,[0,1,2]]
      snapshot['vel'] = pos_vel[:,[3,4,5]]
      snapshot['shrink_centre'] = s[i]['shrink_centre']
      s_interp.append(snapshot)

  # Fix the centring:
  #### NEW ####
  #s[i]['pos'] += s[i]['shrink_centre']
  #s[i]['vel'] += s[i]['vcore']

#===============================#

# Only modifies the s_interp group:
#===============================#
arrays = {}
fields = ['mass', 'rdens', 'rg', 'hlr', 'shrink_centre', 'kstara', 'nbound', 'pos', 'vel']
for field in fields:
  arrays[field] = np.array([i[field] for i in s])

# Define a new resolution:
orig_res = np.linspace(s[0]['age'], s[-1]['age'], len(s))
spline_res = np.linspace(s_interp[0]['age'], s_interp[-1]['age'], len(s_interp))

# Interpolate all necessary arrays to new resolution:
for field in fields[:-4]:
  arrays[field] = make_interp_spline(orig_res, arrays[field], k=3)(spline_res)
arrays['pos'] = CubicHermiteSpline(orig_res, arrays['pos'], arrays['vel']*timeunit)(spline_res)
indices = np.floor(np.interp(spline_res, orig_res, np.arange(len(s)))).astype('int')
arrays['kstara'] = arrays['kstara'][indices]
arrays['nbound'] = arrays['nbound'][indices]

# Replace orbit integration of unbound stars with csplines:
for i in range(len(s_interp)):
  for field in fields[:-2]:
    s_interp[i][field] = arrays[field][i]

  # Replace unbound particles with a cspline, which should look more accurate:
  s_interp[i]['pos'][~s_interp[i]['nbound']] = arrays['pos'][i][~s_interp[i]['nbound']]

  # Un-centre the star cluster (don't bother with the velocities):
  s_interp[i]['pos'] += s_interp[i]['shrink_centre']
#===============================#

#------------------------------------------------------------
