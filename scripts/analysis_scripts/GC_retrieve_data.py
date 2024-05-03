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
import os

from scipy.interpolate import BSpline, make_interp_spline
from scipy.ndimage import uniform_filter
from scipy.interpolate import splrep, BSpline

# Find all the simulation directories:
#------------------------------------------------------------
sims = glob(path + f'Nbody6_sims/{suite}/Halo*')
#data = np.genfromtxt(path + f'scripts/files/GC_property_table_CHIMERA_massive.txt', unpack=True, skip_header=2, dtype=None)

# Load Ethan's property dict:
data = load_data()

def get_dict(key):
  return np.concatenate([[data[i][j][key] for j in list(data[i].keys())] for i in list(data.keys())])

#GC_ID = np.concatenate([[j for j in list(data[i].keys())] for i in list(data.keys())])
GC_ID = get_dict('Internal ID')
GC_birthtime = get_dict('Median birthtime') # [Myr]

lum_max = 100 # Lsol
overwrite = True
#------------------------------------------------------------

# Load the property dictionary:
#------------------------------------------------------------
filename = path + f'scripts/files/GC_data_{suite}.pk1'
if os.path.isfile(filename):
  with open(filename, 'rb') as f:
    GC_data = pickle.load(f)
else:
  GC_data = {}
#------------------------------------------------------------

# Loop over each simulation:
#------------------------------------------------------------
#sims = ['/vol/ph/astro_data/shared/morkney2/GC_mock_obs/Nbody6_sims/Full/']
#sims = ['/vol/ph/astro_data/shared/morkney2/GC_mock_obs/Nbody6_sims/Full/Halo1459_fiducial_hires_output_00020_2']
for sim in sims:

  print(sim)

  ID = int(''.join(c for c in sim.split('_')[-1] if c.isdigit()))
  birthtime = GC_birthtime[np.where(GC_ID == ID)[0][0]]

  # Various accounting:
  if sim.split('/')[-1] not in list(GC_data.keys()):
    GC_data[sim.split('/')[-1]] = {}
    print('>    Creating new data entry.')
  elif not len(GC_data[sim.split('/')[-1]]):
    print('>    Creating new data entry.')
  elif (sim.split('/')[-1] in list(GC_data.keys())) & \
     (GC_data[sim.split('/')[-1]]['t'][-1] <= (13.8+birthtime)) & \
     (GC_data[sim.split('/')[-1]]['mass'][-1] >= 1e3):
    print('>    Updating data entry on %s...' % sim.split('/')[-1])
    pass
  elif (sim.split('/')[-1] in list(GC_data.keys())) & (not overwrite):
    print('>    Data entry already exists.')
    continue

  time_step1 = time.time()
  try:
    s = read_nbody6(sim, df=True)
  except:
    print('>    Data entry has no outputs.')
    continue
  sim = sim.split('/')[-1]

  # Load the starting time:
  print('>    %i' % ID, end='')

  # Initialise property arrays:
  GC_properties = ['t', 'rg', 'vg', 'cum_orb', 'rp', 'ra', 'ecc', 'mass', 'mtot', 'hlr', 'hmr', 'mV', 'mV_hlr', 'mV_hmr']
  for GC_property in GC_properties:
    GC_data[sim][GC_property] = np.ones(len(s)) * np.nan
  GC_properties = ['posg', 'velg']
  for GC_property in GC_properties:
    GC_data[sim][GC_property] = np.ones([len(s),3]) * np.nan

  for i in range(len(s)):

    if GC_data[sim]['t'][i] == (birthtime + s[i]['age']) / 1e3:
      continue

    # Get the simulation time:
    GC_data[sim]['t'][i] = (birthtime + s[i]['age']) / 1e3 # Gyr

    # Centre the GC position:
    body_noBHs = s[i]['nbound'] & (s[i]['kstara'] != 14)
    if not np.sum(body_noBHs):
      continue
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
    GC_data[sim]['mtot'][i] = s[i]['mass'].sum()

    # Calculate the GC size:
    try:
      GC_data[sim]['hlr'][i] = func.R_half(s[i], type='lum', filt=[0., 100.])
    except:
      GC_data[sim]['hlr'][i] = 0.
    try:
      GC_data[sim]['hmr'][i] = func.R_half(s[i], type='mass', filt=[0., 100.])
    except:
      GC_data[sim]['hmr'][i] = 0.

    # Calculate the GC V-band magnitude:
    s[i]['vlum'] = func.Lbol_Lv(s[i]['lum'], s[i]['teff'])
    vcut = s[i]['vlum'] < lum_max
    vlum = s[i]['lum'][body_noBHs * vcut].sum()
    GC_data[sim]['mV'][i] = 4.83 - 2.5*np.log10(vlum)
    vlum_hlr = s[i]['lum'][body_noBHs * vcut * (s[i]['r2']<GC_data[sim]['hlr'][i])].sum()
    area_hlr = 4*np.pi*GC_data[sim]['hlr'][i]**2
    GC_data[sim]['mV_hlr'][i] = (4.83 + 21.572 - 2.5*np.log10(vlum_hlr/area_hlr))
    vlum_hmr = s[i]['lum'][body_noBHs * vcut * (s[i]['r2']<GC_data[sim]['hmr'][i])].sum()
    area_hmr = 4*np.pi*GC_data[sim]['hmr'][i]**2
    GC_data[sim]['mV_hmr'][i] = (4.83 + 21.572 - 2.5*np.log10(vlum_hmr/area_hmr))

  time_step2 = time.time()
  print(', %.2fs' % (time_step2-time_step1))

  # Fit a spline to better-resolve the full orbit:
  not_nan = ~np.any(np.isnan(GC_data[sim]['posg']), axis=1) & (np.gradient(GC_data[sim]['t']) != 0)
  factor = 5
  k = min(np.sum(not_nan)-1, 3)
  orig_res = np.linspace(0, 1, np.sum(not_nan))
  spline_res = np.linspace(0, 1, np.sum(not_nan)*factor)
  # Time should not be a spline to avoid negative values:
  torb = np.interp(spline_res, orig_res, GC_data[sim]['t'][not_nan])
  # Rg must be very smooth to correctly identify apos/peris:
  rorb = np.linalg.norm([BSpline(*splrep(orig_res, uniform_filter(j, size=4), s=len(j)))(spline_res) for j in GC_data[sim]['posg'][not_nan].T], axis=0)

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

    # Also calculate the orbital peri/apo/eccentricity at all times:
    for rap, rap_IDs in zip(['ra', 'rp'], [apos, peris]):
      segments = np.digitize(GC_data[sim]['t'], torb[rap_IDs])
      segments[segments==len(rap_IDs)] -= 1
      GC_data[sim][rap] = rorb[rap_IDs][segments]

    GC_data[sim]['ecc'] = (GC_data[sim]['ra']-GC_data[sim]['rp'])/\
                          (GC_data[sim]['ra']+GC_data[sim]['rp'])

  print('Saving...')
  with open(filename, 'wb') as f:
    pickle.dump(GC_data, f)
#------------------------------------------------------------

