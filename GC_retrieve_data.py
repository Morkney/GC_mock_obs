from config import *
import numpy as np
from read_out3 import read_nbody6
import GC_functions as func

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

from glob import glob
import pickle
import time

# Find all the simulation directories:
#------------------------------------------------------------
sims = glob(path + '/Halo*')
data = np.genfromtxt('./GC_property_table.txt', unpack=True, skip_header=2, dtype=None, delimiter='\t\t\t')
GC_ID = np.array([i[15] for i in data])
GC_birthtime = np.array([i[8] for i in data]) # Myr

lum_max = 100 # Lsol
overwrite = True
#------------------------------------------------------------

# Load the property dictionary:
#------------------------------------------------------------
with open('./GC_data.pk1', 'rb') as f:
  GC_data = pickle.load(f)
#------------------------------------------------------------

# Loop over each simulation:
#------------------------------------------------------------
for sim in sims:

  if (sim in list(GC_data.keys())) & (not overwrite):
    continue

  # Load the starting time:
  ID = int(''.join(c for c in sim.split('_')[-1] if c.isdigit()))
  print('>    %i' % ID, end='')
  birthtime = GC_birthtime[np.where(GC_ID == ID)[0][0]]

  time_step1 = time.time()
  s = read_nbody6(sim, df=True)

  t = np.empty(len(s))
  r = np.empty(len(s))
  mass = np.empty(len(s))
  hlr = np.empty(len(s))
  mV_hlr = np.empty(len(s))

  for i in range(len(s)):

    # Get the simulation time:
    t[i] = (birthtime + s[i]['age']) / 1e3 # Gyr [?]

    # Centre the GC position:
    body_noBHs = s[i]['nbound'] & (s[i]['kstara'] != 14)
    #cen = np.average(s[i]['pos'][body_noBHs], weights=s[i]['mass'][body_noBHs], axis=0)
    #s[i]['pos'] -= cen
    cen = func.shrink(s[i])
    s[i]['pos'] -= cen
    s[i]['r'] = np.linalg.norm(s[i]['pos'], axis=1)
    s[i]['r2'] = np.linalg.norm(s[i]['pos'][:,[0,1]], axis=1)

    # Get the orbital radius:
    r[i] = np.linalg.norm(s[i]['rdens'] + s[i]['rg']*1e3)

    # Calculate the GC mass:
    mass[i] = s[i]['mass'][s[i]['nbound']].sum()

    # Calculate the GC size:
    hlr[i] = func.R_half(s[i])

    # Calculate the GC V-band magnitude:
    s[i]['vlum'] = func.Lbol_Lv(s[i]['lum'], s[i]['teff'])
    vcut = s[i]['vlum'] < lum_max
    central_vlum = s[i]['lum'][body_noBHs * vcut * (s[i]['r2']<hlr[i])].sum()
    central_area = 4*np.pi*hlr[i]**2
    mV_hlr[i] = (4.83 + 21.572 - 2.5*np.log10(central_vlum/central_area))

  time_step2 = time.time()
  print(', %.2fs' % (time_step2-time_step1))

  GC_data[sim] = {}
  GC_data[sim]['t'] = t
  GC_data[sim]['mass'] = mass
  GC_data[sim]['hlr'] = hlr
  GC_data[sim]['mV_hlr'] = mV_hlr
  with open('./GC_data.pk1', 'wb') as f:
    pickle.dump(GC_data, f)
#------------------------------------------------------------
