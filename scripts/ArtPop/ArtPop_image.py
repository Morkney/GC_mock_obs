from config import *

import numpy as np
from read_out3 import read_nbody6
import GC_functions as func
import plot_Nbody6_data

# Astro modules:
import artpop
from astropy import units as u
from astropy.visualization import make_lupton_rgb
from astropy.io import fits

# Pyplot modules:
import default_setup
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt, matplotlib.patches as patches
plt.ion()

# Load chosen simulation:
#------------------------------------------------------------
sim_name = 'Halo1459_fiducial_hires_output_00023_3'
SC_age = 13.8 # [Gyr]

plot_GC = True
centre = 'shrink'
distance_to_GC = 30 # [kpc]
image_size = 501 # [Pixels]
arcsec_per_pixel = 0.05
stretch = 0.2
Q = 8

data = np.genfromtxt(path + 'files/GC_property_table.txt', unpack=True, skip_header=2, dtype=None)
GC_ID = np.where([int(sim_name.split('_')[-1]) == i[15] for i in data])[0][0]
GC_Z = data[GC_ID][7]
GC_birthtime = data[GC_ID][8]

# Sanitise data:
GC_Z = max(min(GC_Z, 0.5), -3.)

rng = 100
#------------------------------------------------------------

# Load the simulation data:
#--------------------------------------------------------------------------
sim = read_nbody6(path+sim_name, df=True)
print('>    Nbody6 simulation %s has been loaded.' % sim_name)

# Select the relevant snapshot:
time = np.array([s['age'] for s in sim])
snapshot = np.abs(time - (SC_age*1e3-GC_birthtime)).argmin()
s = sim[snapshot]
print('>    Snapshot %i with age %.2f Myr' % (snapshot, s['age']))
#--------------------------------------------------------------------------

# Track the full orbit:
#--------------------------------------------------------------------------
orbit_pos = np.empty([len(sim), 3])
for i, s_i in enumerate(sim):

  # Centre the GC position:
  body_noBHs = s_i['nbound'] & (s_i['kstara'] != 14)
  cen = np.average(s_i['pos'][body_noBHs], weights=s_i['mass'][body_noBHs], axis=0)

  # Orbital position:
  orbit_pos[i] = s_i['rdens'] + s_i['rg']*1e3
#--------------------------------------------------------------------------

# Inspect snapshot data:
#--------------------------------------------------------------------------
# Find tail stars, body stars, and black hole type stars:
s['tail'] = ~s['nbound'] & (s['mass'] > 0)
s['body'] = s['nbound'] & (s['mass'] > 0)
s['BHs'] = np.array([kstara in [13, 14] for kstara in s['kstara']])
s['remnants'] = np.array([kstara in [10, 11, 12, 13, 14, 15] for kstara in s['kstara']])

# Centre:
s['pos'] -= s['rdens']
if centre == 'CoM':
  cen = np.average(s['pos'][s['body'] & ~s['BHs']], axis=0, weights=s['mass'][s['body'] & ~s['BHs']])
elif centre == 'shrink':
  cen = func.shrink(s)
else:
  raise Exception('Centre with either CoM or shrink.')
s['pos'] -= cen
s['r'] = np.linalg.norm(s['pos'], axis=1)
print('>    Centred on the GC body.')

# Make a plot:
if plot_GC:
  plot_Nbody6_data.plot_Nbody6(s, sim_name)

hlr = func.R_half(s)

# Remove remnants from arrays that will be accessed by ArtPop:
for field in ['mass', 'pos']:
  s[field] = s[field][~s['remnants']]
#--------------------------------------------------------------------------

# Create source object from Nbody6 data:
# To see all the filters and bands: artpop.phot_system_lookup()
# /user/HS301/m18366/.local/lib/python3.6/site-packages/artpop/
#------------------------------------------------------------------
src = artpop.MISTNbodySSP(
    log_age = np.log10((s['age']+GC_birthtime)*1e6),
    feh = GC_Z,
    r_eff = hlr * u.pc,
    num_r_eff = 2,
    use_stars = s,
    phot_system = 'HST_ACSWF',
    distance = distance_to_GC * u.kpc,
    xy_dim = image_size,
    pixel_scale = arcsec_per_pixel * u.arcsec/u.pixel,
    random_state = rng,)
#------------------------------------------------------------------

# Initialise an Imager object.
#------------------------------------------------------------------
imager = artpop.ArtImager(
    phot_system = 'HST_ACSWF', # photometric system
    diameter = 2.4 * u.m,      # effective aperture diameter
    read_noise = 3,            # read noise in electrons
    random_state = rng)
#------------------------------------------------------------------

# Create mock images:
#------------------------------------------------------------------
# Scale the bands to improve aesthetics:
bands = ['ACS_WFC_F814W', 'ACS_WFC_F606W', 'ACS_WFC_F475W']
scale = dict(ACS_WFC_F814W=1, ACS_WFC_F606W=1, ACS_WFC_F475W=1)
exptime = dict(ACS_WFC_F814W=12830 * u.s, ACS_WFC_F606W=12830 * u.s, ACS_WFC_F475W=12830 * u.s)

# Mock observe:
images = []
for b in bands:
  _psf = fits.getdata(path + f'files/{b}.fits', ignore_missing_end=True)
  obs = imager.observe(src, b, exptime=exptime[b], psf=_psf, zpt=22)
  images.append(obs.image * scale[b])

rgb = make_lupton_rgb(*images, stretch=stretch, Q=Q)
#------------------------------------------------------------------

# Create plot:
#------------------------------------------------------------------
#fig, ax = artpop.show_image(rgb)
fig, ax = plt.subplots(figsize=(6,6))
ims = ax.imshow(rgb, origin='lower')
ax.axis('off')
ax.set_xticks([])
ax.set_yticks([])

# Distance bar:

# Circular crop:
radius = np.shape(rgb)[0]/2. - 1
origin = int(np.shape(rgb)[0]/2.)
patch = patches.Circle((origin, origin), radius=radius, transform=ax.transData)
ims.set_clip_path(patch)
#------------------------------------------------------------------
