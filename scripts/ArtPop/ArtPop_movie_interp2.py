# Personal modules:
from config import *
from read_out3 import read_nbody6
import GC_functions as func

# Pyplot modules:
import default_setup
import matplotlib
#matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import matplotlib.patheffects as path_effects

# Maths modules:
import numpy as np
from scipy.interpolate import interpn, BSpline, make_interp_spline, CubicHermiteSpline
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
from astropy.io import fits
import agama

# Load chosen simulation:
#------------------------------------------------------------
# Timestep for interpolation:
t_min = 1410-929.2797161904183 # [Myr]
t_max = 1430-929.2797161904183 # [Myr]
delta_t = 1 # [Myr]

suite = 'enhanced_mass_suite'
#suite = ''
sim = 'Halo1459_fiducial_hires_output_00023_3'
s = read_nbody6(path+suite+'/'+sim, df=True)

with open('./files/GC_data_%s.pk1' % suite, 'rb') as f:
  GC_data = pickle.load(f)

data = np.genfromtxt('./files/GC_property_table.txt', unpack=True, skip_header=2, dtype=None)
GC_ID = np.where([int(sim.split('_')[-1]) == i[15] for i in data])[0][0]
GC_Z = data[GC_ID][7]
GC_birthtime = data[GC_ID][8]

# Sanitise data:
GC_Z = max(min(GC_Z, 0.5), -3.)

rng = 100
#------------------------------------------------------------

# Various setups:
#------------------------------------------------------------
fs = 10

# Window sizes:
follow = True
params = dict(GC_width=20., GC_pad=10.)
params['full_width'] = GC_data[sim]['rg'].max() + params['GC_width'] + params['GC_pad']

# Make temporary frame storage:
if not os.path.isdir('./pngs'):
  os.mkdir('./pngs')
else:
  os.system('rm -rf ./pngs')
  os.mkdir('./pngs')

# Calculate rotation matrix to place the GC orbit in the x-y plane:
rotation = func.align_orbit(GC_data[sim]['posg'][0], GC_data[sim]['velg'][0])

# Distance bar scales:
for panel in ['full', 'GC']:
  params['%s_ruler' % panel] = int('%.0f' % float('%.1g' % (params['%s_width' % panel]/2.1)))
  params['%s_corner1' % panel] = params['%s_width' % panel] - (0.1 * params['%s_width' % panel]) - params['%s_ruler' % panel]
  params['%s_corner2' % panel] = 0.9 * params['%s_width' % panel]
  params['%s_cap' % panel] = 0.025 * params['%s_width' % panel]

# Scale the bands to improve aesthetics:
bands = ['ACS_WFC_F814W', 'ACS_WFC_F606W', 'ACS_WFC_F475W']
scale = dict(ACS_WFC_F814W=1, ACS_WFC_F606W=1, ACS_WFC_F475W=1)
exptime = dict(ACS_WFC_F814W=12830 * u.s, ACS_WFC_F606W=12830 * u.s, ACS_WFC_F475W=12830 * u.s)
psf = {}
for b in bands:
  psf[b] = fits.getdata(f'./files/{b}.fits', ignore_missing_end=True)

paths = [path_effects.Stroke(linewidth=2, foreground='k'), path_effects.Normal()]

# ArtPop imager:
imager = artpop.ArtImager(
  phot_system = 'HST_ACSWF', # photometric system
  diameter = 2.4 * u.m,      # effective aperture diameter
  read_noise = 3,            # read noise in electrons
  random_state = rng)

# Scaling fractions that don't break the psf symmetries, and associated stretch values:
fractions = np.array([1.0,2.0,2.6,3.1,3.8,4.1,4.6,4.9,5.8,6.2,6.9,7.2,8.6,10.1,13.,16.1,24.])
stretches = 10**(np.log10(fractions)*2.15266 - 0.69897)
#------------------------------------------------------------

# Interpolate the simulation data with a cubic spline:
#------------------------------------------------------------
def sigmoid(x, beta=3.):
  return 1 / (1 + (x / (1-x))**(-beta))
A = sigmoid(np.linspace(0, 1, timesteps))

# Take a time-slice of the simulation data:
times = np.array([i['age'] for i in s])
time_slice = (times >= t_min) & (times <= t_max)
s = [s[i] for i in np.where(time_slice)[0]]

# Define the new set of timesteps using delta_t:
cadence_multiplier = round((s[1]['age']-s[0]['age']) / delta_t)
timesteps = cadence_multiplier + 1
total_time = s[1]['age'] - s[0]['age']

# Sort arrays and ensure all arrays are the same size:
for field in ['mass', 'pos', 'vel', 'kstara', 'nbound']:
  for i in range(len(s)):
    s[i][field] = s[i][field][np.argsort(s[i]['ids'])]
    s[i][field] = s[i][field][:len(s[0][field])]

# Create a new higher-resolution simulation instance:
s_interp = []
print('Integrating orbits...')
agama.setUnits(mass=1, length=1e-3, velocity=1)
timeunit = 0.97779222
for i in range(len(s)):
  # Calculate bulk properties and centre the GC:
  s[i]['shrink_centre'] = func.shrink(s[i])
  s[i]['pos'] -= s[i]['shrink_centre']
  s[i]['r'] = np.linalg.norm(s[i]['pos'], axis=1)
  s[i]['hlr'] = func.R_half(s[i], type='mass', filt=[0., 100.])
  s[i]['vcore'] = np.average(s[i]['vel'][s[i]['r'] <= np.percentile(s[i]['r'][s[i]['nbound']], 30)], axis=0)
  s[i]['vel'] -= s[i]['vcore']

  # Get an Agama profile fit:
  rmax = s[i]['r'][s[i]['nbound']].max()
  data_snap = (np.concatenate([s[i]['pos'], s[i]['vel']], axis=1), s[i]['mass'])
  potential = agama.Potential(type='Multipole', particles=data_snap, gridSizeR=100, rmin=1e-3, rmax=rmax, symmetry='Spherical')

  # Integrate the orbits over the backward timestep and interpolate:
  if i > 0:
    orbits = agama.orbit(ic=data_snap[0], potential=potential, time=-total_time, trajsize=timesteps)
    for j in range(1, timesteps-1):
      pos_vel = np.vstack([k[1][j] for k in orbits])
      s_interp[-j]['pos'] = (1-A[j])*pos_vel[:,[0,1,2]] + A[j]*s_interp[-j]['pos']
      s_interp[-j]['vel'] = (1-A[j])*pos_vel[:,[3,4,5]] + A[j]*s_interp[-j]['vel']

  # Integrate the orbits over the forward timestep:
  if i < len(s) - 1:
    orbits = agama.orbit(ic=data_snap[0], potential=potential, time=total_time, trajsize=timesteps)
    for j in range(timesteps-1):
      snapshot = {}
      snapshot['age'] = s[i]['age'] + orbits[0][0][j]
      pos_vel = np.vstack([k[1][j] for k in orbits])
      snapshot['pos'] = pos_vel[:,[0,1,2]]
      snapshot['vel'] = pos_vel[:,[3,4,5]]
      snapshot['shrink_centre'] = s[i]['shrink_centre']
      s_interp.append(snapshot)
  else:
    snapshot = {}
    snapshot['age'] = s[i]['age']
    snapshot['pos'] = s[i]['pos']
    snapshot['vel'] = s[i]['vel']
    snapshot['shrink_centre'] = s[i]['shrink_centre']
    s_interp.append(snapshot)

print('...Done.')

# Store the snapshot data in arrays:
arrays = {}
fields = ['mass', 'rdens', 'rg', 'hlr', 'shrink_centre', 'kstara', 'nbound', 'pos', 'vel']
for field in fields:
  arrays[field] = np.array([i[field] for i in s])

# Define a new resolution:
orig_res = np.linspace(s[0]['age'], s[-1]['age'], len(s))
spline_res = np.linspace(s[0]['age'], s[-1]['age'], len(s_interp))

# Interpolate all necessary arrays to new resolution:
for field in fields[:-4]:
  arrays[field] = make_interp_spline(orig_res, arrays[field], k=3)(spline_res)
arrays['pos'] = CubicHermiteSpline(orig_res, arrays['pos'], arrays['vel'])(spline_res)
indices = np.floor(np.interp(spline_res, orig_res, np.arange(len(s)))).astype('int')
arrays['kstara'] = arrays['kstara'][indices]
arrays['nbound'] = arrays['nbound'][indices]

# Replace orbit integration of unbound stars with csplines:
print('Justifying...')
for i in range(len(s_interp)):
  for field in fields[:-2]:
    s_interp[i][field] = arrays[field][i]

  # Replace unbound particles with a cspline, which should look more accurate:
  s_interp[i]['pos'][~s_interp[i]['nbound']] = arrays['pos'][i][~s_interp[i]['nbound']]

  # Un-centre:
  s_interp[i]['pos'] += s_interp[i]['shrink_centre']
print('...Done.')

s = s_interp
del s_interp

# Initialise trace:
trace = np.array([])
trace_fade = int(5e1 / s[1]['age'])

# Check that the velocities are actually correct:
#------------------------------------------------------------

# Function to plot a single Nbody6 simulation snapshot frame:
#------------------------------------------------------------
def frame(i):

  print('>    Snapshot %i ' % i)
  if s[i]['nbound'].sum() == 0:
    return

  # Gather metadata:
  string = r'Time$ = %.2f\,{\rm Gyr}$' % ((s[i]['age']+GC_birthtime) / 1e3) + '\n' + \
           r'Mass$ = %s\,$M$_{\odot}$' % func.latex_float(s[i]['mass'][s[i]['nbound']].sum()) + '\n' + \
           r'$R_{\rm half} = %.2f\,{\rm pc}$' % s[i]['hlr']

  fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=1, gridspec_kw={'wspace':0.05})

  # SC-centric and G-centric coordinates:
  pos_SC = np.dot(s[i]['pos'] - s[i]['shrink_centre'], rotation.T)
  pos_G = np.dot(s[i]['pos'] + s[i]['rg']*1e3, rotation.T)

  # Track the orbit:
  trace = np.vstack([s[j]['rdens'] + s[j]['rg']*1e3 for j in range(np.max([0, 1+i-trace_fade]), i+1)])
  trace = np.dot(trace, rotation.T)
  k = max(0, min(3, len(trace)-1))
  orig_res = np.linspace(0, 1, len(trace))
  spline_res = np.linspace(0, 1, len(trace) * 5)
  trace = make_interp_spline(orig_res, trace, k=k)(spline_res)

  # Get GC velocity vector and rotation:
  if follow:
    if i > 0:
      vec = np.array([trace[-1][0] - trace[-2][0], trace[-1][1] - trace[-2][1]])
      theta_xy = func.angle(vec, [1.,0.]) * np.sign(-vec[1])
    else:
      vec = [1, 1, 1]
      theta_xy = 0
    zoom_rotation = func.R_z(theta_xy)

    # Rotations:
    pos_SC = np.dot(pos_SC, zoom_rotation.T)

  # Build the plot:
  #-----------------------------------------------------------------------------
  # Make ArtPop image:
  s[i]['remnants'] = np.array([kstara in [10, 11, 12, 13, 14, 15] for kstara in s[i]['kstara']])
  s[i]['mass'] = s[i]['mass'][~s[i]['remnants']]

  # Select rescaling:
  xy_dims = 501/(params['GC_width']/params['full_width'])/fractions
  choice = np.argmin(np.abs(xy_dims - 1000))
  fraction_scaled = fractions[choice]
  stretch_scaled = stretches[choice]

  for j, (pos, width, stretch, fraction) in enumerate(zip([pos_G, pos_SC], [params['full_width'], params['GC_width']], [stretch_scaled,0.2], [fraction_scaled, 1])):
    s[i]['pos'] = pos[~s[i]['remnants']]
    zoom_amount = params['GC_width']/width
    xy_dim = int(501/zoom_amount/fraction)
    if not xy_dim%2:
      xy_dim += 1

    # ArtPop source:
    src = artpop.MISTNbodySSP(
      log_age = np.log10((s[i]['age']+GC_birthtime)*1e6),
      feh = GC_Z,
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
    for b in bands:
      if j:
        _psf = psf[b]
      else:
        _psf = zoom(psf[b], zoom=1/fraction, order=1)
        _psf *= np.shape(psf[b])[0]**2 / np.shape(_psf)[0]**2
      obs = imager.observe(src, b, exptime=exptime[b], psf=_psf, zpt=22)
      images.append(obs.image * scale[b])
    rgb = make_lupton_rgb(*images, stretch=stretch, Q=8)

    # Add this to the plot:
    ax[j].imshow(rgb, origin='lower', extent=np.array([-1,1,-1,1])*width)

  # Add orbital tracer line:
  line_colour = (np.ones([trace_fade*5, 4]) * np.vstack(np.arange(trace_fade*5)))[-len(trace):] / (trace_fade*5)
  line_colour[:,:3] = 1.
  trace_line = LineCollection([np.column_stack([[trace[i,0], trace[i+1,0]], [trace[i,1], trace[i+1,1]]]) for i in range(len(trace)-1)], \
                              linewidths=1, capstyle='butt', color=line_colour, joinstyle='round')
  ax[0].add_collection(trace_line)

  # Add a box that follows the GC bound body:
  square = patches.Rectangle((trace[-1][0] - params['GC_width'], trace[-1][1] - params['GC_width']), \
                             width=params['GC_width']*2, height=params['GC_width']*2, \
                             facecolor='None', edgecolor='w', lw=1, ls='--')
  if follow:
    square_rotate = mpl.transforms.Affine2D().rotate_deg_around(\
                    trace[-1][0], trace[-1][1], -theta_xy * 180./np.pi) + ax[0].transData
    square.set_transform(square_rotate)
  ax[0].add_patch(square)

  # Add a marker to indicate the centre of the galactic potential:
  ax[0].plot([0,0], [0,0], 'w*', markersize=12, markeredgewidth=1, markeredgecolor='k')

  # Add the metadata:
  ax[0].text(0.025, 0.975, string, va='top', ha='left', color='w', \
             fontsize=fs-2, transform=ax[0].transAxes, path_effects=paths, fontname='monospace')

  # Distance bars:
  for j, (panel, label) in enumerate(zip(['full', 'GC'], ['Orbital plane', 'Zoom-in'])):
    for lw, color, order, capstyle in zip([3,1], ['k', 'w'], [100, 101], ['projecting', 'butt']):
      _, _, cap = ax[j].errorbar([params['%s_corner1' % panel], params['%s_corner1' % panel]+params['%s_ruler' % panel]], \
                                  np.ones(2)*params['%s_corner2' % panel], yerr=np.ones(2)*params['%s_cap' % panel], \
                                  color=color, linewidth=lw, ecolor=color, elinewidth=lw, zorder=order)
      cap[0].set_capstyle(capstyle)

    # Distance bar labels:
    ax[j].text(params['%s_corner1' % panel] + params['%s_ruler' % panel]/2., \
               params['%s_corner2' % panel] - 0.025*params['%s_width' % panel], \
               r'$%.0f\,$kpc' % params['%s_ruler' % panel], \
               va='top', ha='center', color='w', fontsize=fs-2, path_effects=paths)

    # Panel labels:
    ax[j].text(0.025, 0.025, label, va='bottom', ha='left', color='w', \
               fontsize=fs, transform=ax[j].transAxes, path_effects=paths)

  # Manage the axes:
  for axis, width in zip([0,1], [params['full_width'], params['GC_width']]):
    ax[axis].set_yticks([])
    ax[axis].set_xticks([])
    ax[axis].set_aspect(1)
    ax[axis].set_ylim(-width, width)
    ax[axis].set_xlim(-width, width)

  # Save frame as a compressed png file:
  ram = io.BytesIO()
  plt.savefig(ram, format='png', bbox_inches='tight', dpi=300)
  ram.seek(0)
  img = Image.open(ram)
  img_compressed = img.convert('RGB').convert('P', palette=Image.ADAPTIVE)
  img_compressed.save('./pngs/movie_frame_%05d.png' % i, format='PNG')
  plt.close(fig)

  return
print(1/0)
# Recipes:
#===============================================================================
# Halo1459_fiducial_hires_output_00023_3: Evolve
tasks = np.arange(len(s))
end = len(s)
#===============================================================================

frames = []
n_threads = 8
pool = multiprocessing.Pool(n_threads)
pool.map(frame, tasks)

frames = np.array([imageio.imread('./pngs/movie_frame_%05d.png' % i) for i in range(0, end, 1)])
writer = imageio.get_writer('./images/Movie_%s_%s.mp4' % (sim, suite), fps=24)
for frame in frames:
  writer.append_data(frame)
writer.close()

# Remove temporary frame storage:
#os.system('rm -rf ./pngs')

print('Done.')
