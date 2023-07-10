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

# Create an example sersic distribution:
from astropy.modeling import models, fitting
from astropy.utils.exceptions import AstropyUserWarning

rng = 100
r_eff = 300 # 250
num_r_eff = 10
n = 0.6 # 0.8
ellip = 0.4 # 0.3
theta = 125 # 135
xy_dim = 701
distance = 5 # [Mpc units]
pixel_scale = 0.2 * u.arcsec/u.pixel

# Test data:
src = artpop.MISTSersicSSP(
    log_age = 8.5,        # log of age in years
    feh = -1.5,           # metallicity [Fe/H]
    r_eff = r_eff * u.pc,   # effective radius
    n = n,              # Sersic index
    theta = theta * u.deg,  # position angle
    ellip = ellip,          # ellipticity
    num_stars = 1e6,      # number of stars
    phot_system = 'LSST', # photometric system
    distance = distance * u.Mpc, # distance to system
    xy_dim = xy_dim,         # image dimension
    pixel_scale = pixel_scale,    # pixel scale in arcsec / pixel
    random_state = rng,   # random state for reproducibility
)

# Conclusion: this is tricky to do!
# But I don't want *realistic* sersic fits, I want the equivalent ArtPop sersic fits!
# See if I can't reproduce these fits.

# Finally achieved it, but it was really tricky!
# Need to match these changes exactly before performing this on EDGE data.
# Maybe it's too much of a faff, and it depends on distance too, so a new fit
# would be needed whenever we changed distance?

# Convert to Mpc
r_eff/=1e6


r_pix = np.arctan2(r_eff, distance) * u.radian.to('arcsec') * u.arcsec
r_pix = r_pix.to('pixel', u.pixel_scale(pixel_scale)).value
sample_dim = 2 * np.ceil(r_pix * num_r_eff).astype(int) + 1
x_0, y_0 = sample_dim//2, sample_dim//2

shift = (sample_dim - xy_dim) // 2
x = src.x + shift
y = src.y + shift
xy_bins = np.arange(sample_dim)
z = np.histogram2d(x, y, bins=xy_bins)[0]
yy, xx = np.meshgrid(np.arange(sample_dim-1),
                     np.arange(sample_dim-1))
z /= r_pix # ?

# Retrieve sersic values from this:
model = models.Sersic2D(x_0=x_0, y_0=y_0, amplitude=1, r_eff=r_pix, n=n, ellip=ellip, theta=theta*np.pi/180.)
#model2 = models.Sersic2D(x_0=x_0-5, y_0=y_0+5, amplitude=0.9, r_eff=r_pix*1.2, n=n*1.3, ellip=ellip*0.8, theta=theta*np.pi/180.)
model2 = models.Sersic2D(x_0=x_0-5, y_0=y_0+5, amplitude=0.9, r_eff=r_pix*1.2, n=n*1.3, \
                         bounds={'theta':(-np.pi, np.pi), 'ellip':(0, 1)})
model_fit = models.Sersic2D(bounds={'theta':(-np.pi, np.pi), 'ellip':(0, 1)})
fit = fitting.LevMarLSQFitter()
p = fit(model_fit, xx, yy, model(xx, yy))
p = fit(model_fit, xx, yy, z)

# Doesn't matter if the amplitude in the z array is wrong, as long as it is reasonable.
# Initial guesses must be very good.

# Is it failing because there is literally zero data across most of the array?

x = src.x * (r_eff*1e6*2*num_r_eff / xy_dim) - (r_eff*1e6*2*num_r_eff/2.)
y = src.y * (r_eff*1e6*2*num_r_eff / xy_dim) - (r_eff*1e6*2*num_r_eff/2.)

xy_bins = np.linspace(-2000, 2000, 100)
z = np.histogram2d(x, y, bins=xy_bins)[0]
yy, xx = np.meshgrid(xy_bins[:-1],
                     xy_bins[:-1])

# This fitting function is terrible!
# Can't make it work at all...
