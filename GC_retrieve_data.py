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

from scipy.interpolate import BSpline, make_interp_spline

# Find all the simulation directories:
#------------------------------------------------------------
sims = glob(path + '/Halo*')
data = np.genfromtxt('./GC_property_table.txt', unpack=True, skip_header=2, dtype=None)
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
  time_step1 = time.time()
  s = read_nbody6(sim, df=True)

  # Load the starting time:
  ID = int(''.join(c for c in sim.split('_')[-1] if c.isdigit()))
  print('>    %i' % ID, end='')
  birthtime = GC_birthtime[np.where(GC_ID == ID)[0][0]]

  # Initialise property arrays:
  GC_properties = ['t', 'rg', 'vg', 'cum_orb', 'mass', 'hlr', 'hmr', 'mV_hlr', 'mV_hmr']
  for GC_property in GC_properties:
    GC_data[sim][GC_property] = np.empty(len(s))
  GC_properties = ['posg', 'velg']
  for GC_property in GC_properties:
    GC_data[sim][GC_property] = np.empty([len(s),3])

  for i in range(len(s)):

    # Get the simulation time:
    GC_data[sim]['t'][i] = (birthtime + s[i]['age']) / 1e3 # Gyr [?]

    # Centre the GC position:
    body_noBHs = s[i]['nbound'] & (s[i]['kstara'] != 14)
    #cen = np.average(s[i]['pos'][body_noBHs], weights=s[i]['mass'][body_noBHs], axis=0)
    #s[i]['pos'] -= cen
    cen = func.shrink(s[i])
    s[i]['pos'] -= cen
    s[i]['r'] = np.linalg.norm(s[i]['pos'], axis=1)
    s[i]['r2'] = np.linalg.norm(s[i]['pos'][:,[0,1]], axis=1)

    # Get the orbital properties:
    GC_data[sim]['posg'][i] = s[i]['rdens'] + s[i]['rg']*1e3
    GC_data[sim]['velg'][i] = s[i]['vcore'] + s[i]['vg']
    GC_data[sim]['rg'][i] = np.linalg.norm(s[i]['rdens'] + s[i]['rg']*1e3)
    GC_data[sim]['vg'][i] = np.linalg.norm(s[i]['vcore'] + s[i]['vg'])

    # Calculate the GC mass:
    GC_data[sim]['mass'][i] = s[i]['mass'][s[i]['nbound']].sum()

    # Calculate the GC size:
    GC_data[sim]['hlr'][i] = func.R_half(s[i], type='lum', filt=[0., 100.])
    GC_data[sim]['hmr'][i] = func.R_half(s[i], type='mass', filt=[0., 100.])

    # Calculate the GC V-band magnitude:
    s[i]['vlum'] = func.Lbol_Lv(s[i]['lum'], s[i]['teff'])
    vcut = s[i]['vlum'] < lum_max
    vlum_hlr = s[i]['lum'][body_noBHs * vcut * (s[i]['r2']<GC_data[sim]['hlr'][i])].sum()
    area_hlr = 4*np.pi*GC_data[sim]['hlr'][i]**2
    GC_data[sim]['mV_hlr'][i] = (4.83 + 21.572 - 2.5*np.log10(vlum_hlr/area_hlr))
    vlum_hmr = s[i]['lum'][body_noBHs * vcut * (s[i]['r2']<GC_data[sim]['hmr'][i])].sum()
    area_hmr = 4*np.pi*GC_data[sim]['hmr'][i]**2
    GC_data[sim]['mV_hmr'][i] = (4.83 + 21.572 - 2.5*np.log10(vlum_hmr/area_hmr))

  time_step2 = time.time()
  print(', %.2fs' % (time_step2-time_step1))

  # Fit a spline to better-resolve the full orbit:
  factor = 5
  k = min(len(s)-1, 3)
  orig_res = np.linspace(0, 1, len(s))
  spline_res = np.linspace(0, 1, len(s)*factor)
  torb = make_interp_spline(orig_res, GC_data[sim]['t'], k=k)(spline_res)
  rorb = np.linalg.norm(make_interp_spline(orig_res, GC_data[sim]['posg'], k=k)(spline_res), axis=1)

  # Find the apo- and pericentres:
  differential = np.diff(np.sign(np.diff(rorb)))
  peris = np.where(differential > 0)[0] + 1
  apos = np.where(differential < 0)[0] + 1
  inflections = np.sort(np.concatenate([apos, peris]))
  orb_count = (np.arange(len(apos) + len(peris)) + 0.0) / 2.

  if len(orb_count) <= 1:
    GC_data[sim]['cum_orb'] *= 0
  else:
    # Extend by one unit to catch the boundary conditions:
    orb_count = np.append(np.append(orb_count[0]-0.5, orb_count), orb_count[-1]+0.5)
    torb = np.append(np.append(torb[0]-np.diff(torb[inflections][:2]), torb), torb[-1]+np.diff(torb[inflections][-2:]))
    rorb = np.append(np.append(rorb[inflections[1]], rorb), rorb[inflections[-2]])

    # interpolate to find the number of orbits:
    cumrorb = np.cumsum(np.abs(np.diff(rorb, prepend=rorb[0])))
    orb_count = np.interp(cumrorb, cumrorb[np.concatenate([[0], inflections, [-1]])], orb_count)
    GC_data[sim]['cum_orb'] = np.interp(GC_data[sim]['t'], torb, orb_count)
    GC_data[sim]['cum_orb'] -= GC_data[sim]['cum_orb'][0]

  with open('./GC_data.pk1', 'wb') as f:
    pickle.dump(GC_data, f)
#------------------------------------------------------------
