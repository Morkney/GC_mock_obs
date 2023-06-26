from config import *

import numpy as np
import os
from glob import glob

import astropy
from astropy.io import fits
#from astropy.utils.data import get_pkg_data_filename

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt, matplotlib.patches as patches
plt.ion()

'''
files = glob(path + sim_name + '/fits-files-%s/*' % bands[1])

file = files[-3]
fits.info(file)
image_data = fits.getdata(file, ext=0)

plt.imshow(image_data, norm=LogNorm())
'''

bands = ['V', 'V', 'V']

#------------------------------------------------------------------
#for file in files:
# Create a stacked image of R, G, and B for three different photometric bands:
cimg = [''] * len(bands)
for colour, band in enumerate(bands):
  print('>    Band %s' % band)

  files = glob(path + sim_name + '/fits-files-%s/*' % band)
  file = files[0]
  image_data = fits.getdata(file, ext=0)
  cimg[colour] = image_data

# Turn into an image array:
def normalise(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Not sure if .T is needed or not:
cimg = (normalise(np.array([*cimg]).T) * 254.99).astype(np.uint8)
#------------------------------------------------------------------

# May need to play with the normalisations to produce an aesthetic image:
#------------------------------------------------------------------
#------------------------------------------------------------------

# Create plot:
#------------------------------------------------------------------
fs = 14
fig, ax = plt.subplots(figsize=(6, 6))

ax.imshow(cimg, origin='lower')

#ax.set_xlim(extent[[0,1]])
#ax.set_ylim(extent[[2,3]])
ax.set_xticks([])
ax.set_yticks([])
ax.set_axis_off()

# Remove margins:
fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
ax.margins(0,0)
