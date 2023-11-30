from config import *
import numpy as np
from read_out3 import read_nbody6
import GC_functions as func
import tangos

import default_setup
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
plt.ion()

from glob import glob
import pickle
import time

from scipy.interpolate import BSpline, make_interp_spline

# Load the property dictionary:
#------------------------------------------------------------
with open('./files/GC_data_enhanced_mass_suite.pk1', 'rb') as f:
  GC_data = pickle.load(f)
#------------------------------------------------------------

def normalise(a, normmin, normmax, vmin, vmax):
  normed = (a-vmin) / (vmax-vmin) * (normmax-normmin) + normmin
  normed[normed > normmax] = normmax
  normed[normed < normmin] = normmin
  return normed

mass_ini = []
hmr_ini = []
mass_multiplier = []
hmr_multiplier = []
ts = []
count_ID = []

data = np.genfromtxt('./files/GC_property_table.txt', unpack=True, skip_header=2, dtype=None)
GC_ID = np.array([i[15] for i in data])
GC_hlr = np.array([i[10] for i in data]) # pc
GC_mass = np.array([10**i[9] for i in data]) # Msol

# Loop over each simulation:
for sim in list(GC_data.keys()):

  count_ID.append(int(sim.split('_')[-1]))

  # Locate the time that the GC was recovered from EDGE:
  EDGE_halo = sim.split('/')[-1].split('_')[0]
  EDGE_sim = sim.split('/')[-1].split('_output')[0]
  EDGE_output = sim.split('/')[-1].split('_')[3]
  tangos.core.init_db(TANGOS_path + EDGE_halo + '.db')
  session = tangos.core.get_default_session()
  timestep = int(np.where([EDGE_output in i.extension for i in tangos.get_simulation(EDGE_sim).timesteps])[0])
  EDGE_t = tangos.get_simulation(EDGE_sim).timesteps[timestep].time_gyr

  # Locate the birth time from the simulation:
  GC_t = GC_data[sim]['t'][0]

  # Multiple the mass evolution until the starting mass is at EDGE_t:
  factor = 5
  k = min(len(GC_data[sim]['t'])-1, 3)
  orig_res = np.linspace(0, 1, len(GC_data[sim]['t']))
  spline_res = np.linspace(0, 1, len(GC_data[sim]['t'])*factor)
  mass = make_interp_spline(orig_res, GC_data[sim]['mtot'], k=k)(spline_res)
  hmr = make_interp_spline(orig_res, GC_data[sim]['hmr'], k=k)(spline_res)
  t = make_interp_spline(orig_res, GC_data[sim]['t'], k=k)(spline_res)
  place = np.argmin(np.abs(t - EDGE_t))

  data_place = np.where(int(sim.split('_')[-1]) == GC_ID)

  mass_multiplier.append(GC_mass[data_place] / mass[place])
  hmr_multiplier.append(GC_hlr[data_place] / hmr[place])
  ts.append(EDGE_t-GC_t)
  mass_ini.append(GC_mass[data_place])
  hmr_ini.append(GC_hlr[data_place])
mass_multiplier = np.array(mass_multiplier)
hmr_multiplier = np.array(hmr_multiplier)
ts = np.array(ts)
mass_ini = np.array(mass_ini)
hmr_ini = np.array(hmr_ini)

plt.scatter(ts, mass_multiplier, s=normalise(hmr_ini, 10, 40, 0.12, 12.3), c=normalise(np.log10(mass_ini), 0, 1, np.log10(230), np.log10(27500)))
plt.ylim(0.9, 1.6)
plt.xlim(0., 0.08)

plt.xlabel('EDGE output time - birth time [Gyr]')
plt.ylabel('Mass multiplier')

'''
plt.scatter(ts, hmr_multiplier, s=normalise(hmr_ini, 10, 40, 0.12, 12.3), c=normalise(np.log10(mass_ini), 0, 1, np.log10(230), np.log10(27500)))
plt.ylim(0, 1.1)
plt.xlim(0., 0.08)
plt.xlabel('EDGE output time - birth time [Gyr]')
plt.ylabel('Hmr multiplier')
'''

count_ID = np.array(count_ID)
sorted = np.argsort(count_ID)
count_ID = count_ID[sorted]
mass_multiplier = np.ravel(mass_multiplier[sorted])
hmr_multiplier = np.ravel(hmr_multiplier[sorted])
ts = ts[sorted]
#np.savetxt('./files/GC_multipliers_CHIMERA_massive.txt', np.transpose([count_ID, mass_multiplier, hmr_multiplier, ts]), \
#           fmt='%i\t%f\t%f\t%f', header='Simulation\tMass_mult\thmr_mult\tdelta(t)', delimiter='\t')
