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

# Confirm I can reproduce known result:
# create young, metal poor SSP source at 5 Mpc
src = artpop.MISTSersicSSP(
    log_age = 8.5,        # log of age in years
    feh = -1.5,           # metallicity [Fe/H]
    r_eff = 250 * u.pc,   # effective radius
    n = 0.8,              # Sersic index
    theta = 135 * u.deg,  # position angle
    ellip = 0.3,          # ellipticity
    num_stars = 1e6,      # number of stars
    phot_system = 'LSST', # photometric system
    distance = 5 * u.Mpc, # distance to system
    xy_dim = 701,         # image dimension
    pixel_scale = 0.2,    # pixel scale in arcsec / pixel
    random_state = rng,   # random state for reproducibility
)

use_stars['masses'] = np.ones_like(src.x)
use_stars['num_stars'] = 1e5
use_stars['x'] = src.x * (5000 / 701) - (5000/2.)
use_stars['y'] = src.y * (5000 / 701) - (5000/2.)

src2 = artpop.MISTNbodySSP(
    log_age = 8.5,        # log of age in years
    feh = -1.5,           # metallicity [Fe/H]
    r_eff = 250 * u.pc,   # effective radius
    use_stars = use_stars,
    phot_system = 'LSST', # photometric system
    distance = 5 * u.Mpc, # distance to system
    xy_dim = 701,         # image dimension
    pixel_scale = 0.2,    # pixel scale in arcsec / pixel
    random_state = rng,   # random state for reproducibility
)
difference = (src.x - 700/2.) / (src2.x - 700/2.)
print(difference.mean())
