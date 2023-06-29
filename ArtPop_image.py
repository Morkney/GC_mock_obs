from config import *

import numpy as np
from read_out3 import read_nbody6
import GC_functions as func
import plot_Nbody6_data

import artpop
from astropy import units as u
from astropy.visualization import make_lupton_rgb

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt, matplotlib.patches as patches
plt.ion()

plot_GC = False
centre = 'shrink'

# Tasks for next time:
# 1. Solve the issue of the remnant masses. Do I need to account for this or is it already baked in?
# 3. Read ArtPop paper: https://arxiv.org/pdf/2109.13943.pdf
# 4. Code up extraction of data from EDGE snapshots.
# 5. Make the pipeline run smoothely.

# /user/HS301/m18366/.local/lib/python3.6/site-packages/artpop/

# Load the simulation data:
#--------------------------------------------------------------------------
sim = read_nbody6(path+sim_name, df=True)
print('>    Nbody6 simulation %s has been loaded.' % sim_name)

Z = 0.001 # Set the metallicity

# Select the relevant snapshot:
snapshot = -1
s = sim[snapshot]
age = s['age'] # [Myr]
print('>    Snapshot %i with age %.2f Myr' % (snapshot, age))
#--------------------------------------------------------------------------

# Inspect snapshot data:
#--------------------------------------------------------------------------
# Find tail stars, body stars, and black hole type stars:
s['tail'] = ~s['nbound'] & (s['mass'] > 0)
s['body'] = s['nbound'] & (s['mass'] > 0)
s['BHs'] = s['kstara'] == (13 and 14)

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
  plot_Nbody6_data.plot_Nbody6(s)

rng = 100
#--------------------------------------------------------------------------

# Create source object from Nbody6 data:
# To see all the filters and bands: artpop.phot_system_lookup()
#------------------------------------------------------------------
src = artpop.MISTNbodySSP(
    log_age = np.log10(age * 1e6),
    feh = Z,
    r_eff = 2 * u.pc,
    num_r_eff = 1,
    use_stars = s,
    phot_system = 'LSST',
    distance = 0.1 * u.Mpc,
    xy_dim = 701,
    pixel_scale = 0.1 * u.arcsec/u.pixel,
    random_state = rng,
)
#------------------------------------------------------------------

x = src.x * (2*2*1 / 701) - (2*2*1/2)
print(s['pos'][:,0])
print(x)

print(1/0)

# Initialise an Imager object.
#------------------------------------------------------------------
imager = artpop.ArtImager(
    phot_system = 'LSST', # photometric system
    diameter = 6.4 * u.m, # effective aperture diameter
    read_noise = 4,       # read noise in electrons
    random_state = rng    # random state for reproducibility
)
#------------------------------------------------------------------

# Mock observe the source using the observe method.
#------------------------------------------------------------------
# PSF with 0.6'' seeing
psf = artpop.moffat_psf(fwhm=0.6*u.arcsec)

# observe in gri (assuming the same seeing in all bands)
obs_g = imager.observe(
    source = src,         # source object
    bandpass = 'LSST_g',  # bandpass of observation
    exptime = 15 * u.min, # exposure time
    sky_sb = 22,          # sky surface brightness
    psf = psf             # point spread function
)
obs_r = imager.observe(src, 'LSST_r', 15 * u.min, sky_sb=21, psf=psf)
obs_i = imager.observe(src, 'LSST_i', 30 * u.min, sky_sb=20, psf=psf)
rgb = make_lupton_rgb(obs_i.image, obs_r.image, obs_g.image, stretch=0.2)

# Show image:
artpop.show_image(rgb);
#------------------------------------------------------------------
