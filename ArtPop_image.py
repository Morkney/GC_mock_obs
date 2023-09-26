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

sim_name = 'Halo1459_fiducial_hires_output_00020_1'
SC_age = 1 # [Gyr]
plot_GC = True
centre = 'shrink'
GC_Z = np.genfromtxt('./GC_property_table.txt', unpack=True, skip_header=2, dtype=None)[GC_ID][2]

# /user/HS301/m18366/.local/lib/python3.6/site-packages/artpop/

# Load the simulation data:
#--------------------------------------------------------------------------
sim = read_nbody6(path+sim_name, df=True)
print('>    Nbody6 simulation %s has been loaded.' % sim_name)

# Select the relevant snapshot:
time = np.array([s['age'] for s in sim])
snapshot = np.abs(time - SC_age*1e3).argmin()
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
  orbit_pos[i] = s_i['rdens'] + s_i['rg']*1e3 + cen
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
  plot_Nbody6_data.plot_Nbody6(s)

hlr = func.R_half(s)

rng = 100
#--------------------------------------------------------------------------

# Create source object from Nbody6 data:
# To see all the filters and bands: artpop.phot_system_lookup()
#------------------------------------------------------------------

# Remove remnants from arrays that will be accessed by ArtPop:
for field in ['mass', 'pos']:
  s[field] = s[field][~s['remnants']]

src = artpop.MISTNbodySSP(
    log_age = np.log10(s['age'] * 1e6),
    feh = GC_Z,
    r_eff = hlr * u.pc,
    num_r_eff = 10,
    use_stars = s,
    phot_system = 'LSST',
    distance = 0.2 * u.Mpc,
    xy_dim = 701,
    pixel_scale = 0.2 * u.arcsec/u.pixel,
    random_state = rng,
)
#------------------------------------------------------------------

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
psf = artpop.moffat_psf(fwhm=0.8*u.arcsec)

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

print(1/0)

# Insert a background image:
#------------------------------------------------------------------
# Standard library imports
from copy import deepcopy
from io import BytesIO

# Third-party imports
import requests
from astropy.io import fits

url_prefix = 'https://www.legacysurvey.org/viewer/'

def fetch_psf(ra, dec):
    """
    Returns PSFs in dictionary with keys 'g', 'r', and 'z'.
    """
    url = url_prefix + f'coadd-psf/?ra={ra}&dec={dec}&layer=dr8&bands=grz'
    session = requests.Session()
    resp = session.get(url)
    hdulist = fits.open(BytesIO(resp.content))
    psf = {'grz'[i]: hdulist[i].data for i in range(3)}
    return psf

def fetch_coadd(ra, dec):
    """
    Returns coadds in dictionary with keys 'g', 'r', and 'z'.
    """
    url = url_prefix + f'cutout.fits?ra={ra}&dec={dec}&size=900&'
    url += 'layer=ls-dr8&pixscale=0.262&bands=grz'
    session = requests.Session()
    resp = session.get(url)
    cutout = fits.getdata(BytesIO(resp.content))
    image = {'grz'[i]: cutout[i, :, :] for i in range(3)}
    return image

# random coordinates in Legacy Survey footprint
ra,dec = 182.5002, 12.5554

# grab the model grz PSFs at this location
psf = fetch_psf(ra, dec)

# grab the grz coadds at this location
real_image = fetch_coadd(ra, dec)

# see what a RGB image at this location looks like
rgb = make_lupton_rgb(real_image['z'], real_image['r'],
                      real_image['g'], stretch=0.04)

artpop.show_image(rgb);
#------------------------------------------------------------------
