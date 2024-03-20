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

# Load chosen simulation:
#------------------------------------------------------------
suite = 'fantasy_cores_suite'
suite = 'enhanced_mass_suite'
sim = 'Halo1459_fiducial_hires_output_00023_3'
s = read_nbody6(path+suite+'/'+sim, df=True)
print('GC data loaded...')

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

# Calculate bulk properties of the star cluster:
#------------------------------------------------------------
# Pre-build the parameters:
for i in range(len(s)):
  s[i]['shrink_centre'] = np.nan
  s[i]['r'] = np.nan
  s[i]['hlr'] = np.nan
  s[i]['vcore'] = np.nan

def phase_centre(i):
  for field in ['mass', 'pos', 'vel', 'kstara', 'nbound']:
    i[field] = i[field][np.argsort(i['ids'])]
    i[field] = i[field][:len(s[0][field])]
  i['shrink_centre'] = func.shrink(i)
  i['pos'] -= i['shrink_centre']
  i['r'] = np.linalg.norm(i['pos'], axis=1)
  i['hlr'] = func.R_half(i, type='mass', filt=[0., 100.])
  i['vcore'] = np.average(i['vel'][i['r'] <= np.percentile(i['r'][i['nbound']], 60)], axis=0)
  i['vel'] -= i['vcore']
  return i

n_threads = 16
pool = multiprocessing.Pool(n_threads)
s = pool.map(phase_centre, s)
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
  pos_SC = np.dot(s[i]['pos'], params['rotation'].T)
  pos_G = np.dot(s[i]['pos'] + s[i]['rg']*1e3 + s[i]['shrink_centre'], params['rotation'].T)

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
  plt.close(fig)
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
time_span = 20 # [Myr]
t_min = 0 + GC_birthtime # [Myr]
t_max = 0 + time_span + GC_birthtime # [Myr]
delta_t = 0.5 # [Myr]
s2, begin_len, begin_mult = interpolation.combined_interpolation(s, t_min, t_max, delta_t)
print(1/0)
# Super-high fidelity conclusion:
time_span = 80 # [Myr]
t_min = end_time - time_span # [Myr]
t_max = end_time # [ Myr]
delta_t = 0.5 # [Myr]
s3, end_len, end_mult = interpolation.combined_interpolation(s2, t_min, t_max, delta_t)
print(1/0)
times = np.array([i['age'] for i in s])
end_frame = np.abs(times - end_time).argmin()
#------------------------------------------------------------

# Recipes:
#===============================================================================
fps = 24

# Wind-up period through the initial high-fidelity times:
begin_smooth = np.cumsum(np.arange(begin_mult+1))

# Wind-up period through the concluding high-fidelity times:
end_smooth = np.cumsum(np.arange(end_mult+1))
end_smooth = end_smooth.max()-end_smooth[::-1]

# Wind-up period in the middle times:
middle_smooth = np.cumsum(np.arange(1,30)) # 30
middle_time = 5*fps
end_zoom = end_frame-end_len-middle_smooth.max()
begin_zoom = begin_len+middle_smooth.max()
middle_cadence = int(round((end_zoom - begin_zoom) / middle_time))
middle_cadence = np.diff(middle_smooth).max()

index = np.hstack([np.arange(begin_len - begin_smooth.max()), \
                   begin_smooth + begin_len - begin_smooth.max(), \
                   middle_smooth + begin_len, \
                   np.arange(begin_zoom, end_zoom, middle_cadence)[1:-1], \
                   end_zoom + middle_smooth.max()-middle_smooth[::-1], \
                   end_zoom + middle_smooth.max() + end_smooth, \
                   end_zoom + middle_smooth.max() + end_smooth.max() + np.arange(end_len - end_smooth.max())
                   ])

# Add zoom symbol:
keys = np.concatenate([np.zeros(begin_len - begin_smooth.max()), \
                       np.ones(len(index) - (begin_len - begin_smooth.max()) - (end_len - end_smooth.max())), \
                       np.zeros(end_len - end_smooth.max())]).astype('int')

rotations = np.zeros_like(index)

tasks = np.array([index, rotations, keys]).T
#===============================================================================

# Plotting:
#------------------------------------------------------------
# Make temporary frame storage:
if not os.path.isdir('./pngs2'):
  os.mkdir('./pngs2')
else:
  os.system('rm -rf ./pngs2')
  os.mkdir('./pngs2')

# Loop over all snapshots and make frames:
frames = []
n_threads = 30
pool = multiprocessing.Pool(n_threads)
pool.map(frame, tasks)

# Turn frames into mp4 file:
#frames = np.array([imageio.imread('./pngs2/movie_frame_%05d.png' % i) for i in tasks])
frames = np.array([imageio.imread('./pngs2/%s' % i) for i in np.sort(os.listdir('pngs2'))])
writer = imageio.get_writer('./images/Movie_%s_%s.mp4' % (sim, suite), fps=24)
for frame in frames:
  writer.append_data(frame)
writer.close()

# Remove temporary frame storage:
#os.system('rm -rf ./pngs2')
#------------------------------------------------------------

print('Done.')
