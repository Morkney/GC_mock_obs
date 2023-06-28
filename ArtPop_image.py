from config import *

import artpop
import numpy as np
from astropy import units as u
from astropy.visualization import make_lupton_rgb

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt, matplotlib.patches as patches
plt.ion()

# Tasks for next time:
# 1. Solve the issue of the remnant masses. Do I need to account for this or is it already baked in?
# 2. Solve sample_fraction issue. Might already be OK.
# 3. Read ArtPop paper: https://arxiv.org/pdf/2109.13943.pdf
# 4. Code up extraction of data from EDGE snapshots.
# 5. Make the pipeline run smoothely.

# /user/HS301/m18366/.local/lib/python3.6/site-packages/artpop/

# Step 1: Load the necessary data from Nbody6: x, y, z, L, Teff, m, Z
#------------------------------------------------------------------
file = path + sim_name + '/reduced_data_%s.dat' % sim_name
x,y,z, L, logTeff, m, Z, phase = np.loadtxt(file, skiprows=1, unpack=True)

# REMEMBER TO FILTER NON-VISIBLE STARS!
# WDs, NS, BHs, these are counted internally as remnant mass!

# It appears as though initial stellar masses are constrained to 0.1 < m/Msol < 2.326
# Not sure why this is, but very inconvenient! How can I fix this?
# src.sp.isochrone.isochrone_full['initial_mass'].min()
# src.sp.isochrone.isochrone_full['initial_mass'].max()
m_min = 0.1
m_max = 500. #2.32609933733922
# These come directly from the MIST files for LSST stored at ~/.artpop/mist

use_stars = {}
use_stars['num_stars'] = len(m)
m[m > m_max] = 0.1 #m_max * 0.99
m[m < m_min] = m_min
use_stars['masses'] = m
use_stars['x'] = x
use_stars['y'] = y
Z = Z[0]
rng = 100
#------------------------------------------------------------------

# Create source object from Nbody6 data:
# To see all the filters and bands: artpop.phot_system_lookup()
#------------------------------------------------------------------
src = artpop.MISTNbodySSP(
    log_age = 6.0,
    feh = Z,
    r_eff = 2 * u.pc,
    num_r_eff = 1,
    use_stars = use_stars,
    phot_system = 'HST_ACSWF',
    distance = 0.1 * u.Mpc,
    xy_dim = 701,
    pixel_scale = 0.1 * u.arcsec/u.pixel,
    random_state = rng,
)
#------------------------------------------------------------------

# Initialise an ideal Imager object and plot it.
#------------------------------------------------------------------
'''
imager = artpop.IdealImager()

psf = artpop.moffat_psf(fwhm=0.1*u.arcsec)

obs = imager.observe(src, bandpass='LSST_i', psf=psf, zpt=27)
artpop.show_image(obs.image, figsize=(6,6))
'''
#------------------------------------------------------------------

# Initialise an Imager object.
#------------------------------------------------------------------
imager = artpop.ArtImager(
    phot_system = 'HST_ACSWF', # photometric system
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
    bandpass = 'ACS_WFC_F814W',  # bandpass of observation
    exptime = 15 * u.min, # exposure time
    sky_sb = 22,          # sky surface brightness
    psf = psf             # point spread function
)
obs_r = imager.observe(src, 'ACS_WFC_F606W', 15 * u.min, sky_sb=21, psf=psf)
obs_i = imager.observe(src, 'ACS_WFC_F475W', 30 * u.min, sky_sb=20, psf=psf)
rgb = make_lupton_rgb(obs_i.image, obs_r.image, obs_g.image, stretch=0.4)

# Show image:
artpop.show_image(rgb);
#------------------------------------------------------------------
