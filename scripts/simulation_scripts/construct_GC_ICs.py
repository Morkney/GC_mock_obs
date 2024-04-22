from config import *

import numpy as np
import pynbody
import tangos
import GC_functions as func
import sys
import os

from lmfit import Parameters, Model

import default_setup
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

# Enter EDGE1 to terminal to load pynbody modules

# Simulation choices:
#--------------------------------------------------------------------------
binary_fractions = (0, 0.95)
binary_fraction = binary_fractions[0]
#--------------------------------------------------------------------------

# Parameters:
#--------------------------------------------------------------------------
host_sim = sys.argv[1]
GC_ID = int(sys.argv[2])
plot_fit = False
save_results = True
#--------------------------------------------------------------------------

# Load the GC properties from file:
#--------------------------------------------------------------------------
'''
data = np.genfromtxt(path+'/scripts/files/GC_property_table.txt', unpack=True, skip_header=2, dtype=None)[GC_ID-1]

GC_pos = np.array([data[1], data[2], data[3]]) * 1e3 # pc
GC_vel = np.array([data[4], data[5], data[6]]) # km s^-1
GC_hlr = data[10] # pc
GC_mass = 10**data[9] # Msol
GC_Z = data[7] # dec
GC_birthtime = data[8] # Myr
EDGE_output = 'output_%05d' % data[12]
EDGE_sim_name = data[11].decode("utf-8")
EDGE_halo = int(data[13]) + 1
internal_ID = int(data[14])
count_ID = int(data[15])

print('>    %i' % count_ID)
'''

# Load Ethan's property dict:
data = load_data()

# Parse GC properties:
GC_pos = np.array(data[host_sim][GC_ID]['Galacto-centred position'])
GC_vel = np.array(data[host_sim][GC_ID]['Galacto-centred velocity'])
GC_hlr = data[host_sim][GC_ID]['3D half-mass radius']
GC_mass = data[host_sim][GC_ID]['Stellar Mass'].sum()
GC_Z = data[host_sim][GC_ID]['Median Fe/H']
GC_birthtime = data[host_sim][GC_ID]['Median birthtime']
EDGE_output = 'output_%05d' % data[host_sim][GC_ID]['Output Number']
EDGE_sim_name = host_sim
EDGE_halo = data[host_sim][GC_ID]['Tangos Halo ID'] + 1
internal_ID = data[host_sim][GC_ID]['Internal ID'] # Non-exclusive

# Parse particle properties:
GC_masses = data[host_sim][GC_ID]['Stellar Mass']
GC_metals = data[host_sim][GC_ID]['Fe/H Values']
GC_births = data[host_sim][GC_ID]['Birth Times']
#--------------------------------------------------------------------------

# Load the simulation snapshot:
#--------------------------------------------------------------------------
sim_type = 'CHIMERA' if '383' in EDGE_sim_name else 'EDGE'
EDGE_path = EDGE_path[sim_type]
TANGOS_path = TANGOS_path[sim_type]
tangos.core.init_db(TANGOS_path + EDGE_sim_name.split('_')[0] + '.db')
session = tangos.core.get_default_session()

h = tangos.get_halo(('/').join([EDGE_sim_name, EDGE_output, 'halo_%i' % EDGE_halo]))
if np.linalg.norm(GC_pos)/1e3 > 40.:
  print('The GC position is beyond the R200 radius. Do not simulate.')
  sys.exit(0)

# Get EDGE simulation density, averaged over next 3 simulation snapshots:
r_range = (0.02, 10)
fit_range = (0.035, 3)
if 'CHIMERA' in sim_type:
  data1 = np.loadtxt(path+'/scripts/files/CHIMERA_massive/rho_%s.txt' % h.previous.timestep.extension, unpack=True)
  data2 = np.loadtxt(path+'/scripts/files/CHIMERA_massive/rho_%s.txt' % h.timestep.extension, unpack=True)
  data3 = np.loadtxt(path+'/scripts/files/CHIMERA_massive/rho_%s.txt' % h.next.timestep.extension, unpack=True)
  EDGE_r = np.mean([data1[0], data2[0], data3[0]], axis=0)
  EDGE_rho = np.mean([data1[1], data2[1], data3[1]], axis=0)
else:
  EDGE_r, EDGE_rho = h.previous.calculate_for_descendants('rbins_profile', 'dm_density_profile', nmax=2)
  r = np.logspace(*np.log10(r_range), 100)
  for i in range(3):
    EDGE_r[i], EDGE_rho[i] = func.rebin(EDGE_r[i], EDGE_rho[i], r)
  EDGE_r = EDGE_r[0]
  EDGE_rho = np.mean(EDGE_rho, axis=0)

if np.linalg.norm(GC_pos)/1e3 > fit_range[1]:
  print('GC position is greater that fit_max, %.2f kpc.' % fit_range[1])
elif np.linalg.norm(GC_pos)/1e3 < fit_range[0]:
  print('GC position is less that fit_min, %.2f kpc.' % fit_range[0])
fit_range_arr = (EDGE_r > fit_range[0]) & (EDGE_r <= fit_range[1])

print(f'>    Loaded EDGE density profile for {EDGE_sim_name}, {EDGE_output}.')
#--------------------------------------------------------------------------

# Scale the mass according to the stellar mass evolution:
#--------------------------------------------------------------------------
import stellar_devolution_functions as StellarDevolution

GC_time = h.calculate('t()')
param = StellarDevolution.Parameters('EDGE1')

# Integrate each particle individually:
dt = 0.1 # [Myr]
GC_mass = 0
for i in range(len(GC_masses)):
  stars = StellarDevolution.Stars(npartmax=1)
  stars.add_stars(GC_masses[i], 10**GC_metals[i], GC_time*1e3-GC_births[i])
  for j in range(int(np.round((GC_time*1e3 - GC_births[i]) / dt))):
    stars.evolve(dt, param)
  GC_mass += stars.mass[0]
#--------------------------------------------------------------------------

# Scale the half-light radius if necessary:
#--------------------------------------------------------------------------
if 'compact' in suite:
  # Load the entire array of GC hlr and GC initial mass (adjusted for evolution, too!)
  # Best to calculate these separately and then load them in...
  # In fact, better to do all the calculations in the other script and then load them in...
  pass
#--------------------------------------------------------------------------

# Reconstruct the potential at this time:
#--------------------------------------------------------------------------
def interp(param, alpha):
  return alpha*param[0] + (1-alpha)*param[1]

import pickle
filename = path+'/scripts/files/host_profiles_dict.pk1'
with open(filename, 'rb') as file:
  props = pickle.load(file)

sim = '_'.join([EDGE_sim_name, suite])
time = props[sim]['time']
rs = props[sim]['rs']
gamma = props[sim]['gamma']
Mg = props[sim]['Mg']
i = np.where((GC_birthtime > time[:-1]*1e3) & (GC_birthtime <= time[1:]*1e3))[0][0]

alpha = (GC_birthtime - time[i]*1e3) / (time[i+1]*1e3 - time[i]*1e3)
rs = interp(rs[[i,i+1]], alpha)
gamma = interp(gamma[[i,i+1]], alpha)
#--------------------------------------------------------------------------

# Plot the result and compare:
#--------------------------------------------------------------------------
if plot_fit:
  fs = 14
  fig, ax = plt.subplots(figsize=(6, 6))

  ax.loglog(EDGE_r, EDGE_rho, 'k', lw=2, label='%s, %s' % (EDGE_sim_name, EDGE_output))
  ax.axvline(np.linalg.norm(GC_pos/1e3), c='k', lw=1)

  label = r'Fit %i: ' % 0 + \
          r'$r_{\rm s}=%.2f$, ' % (rs) + \
          r'$M_{\rm g}=%s\,$M$_{\odot}$, ' % func.latex_float(Mg) + \
          r'$\gamma=%.2g$' % gamma
  ax.loglog(EDGE_r, func.Dehnen_profile(EDGE_r, np.log10(rs), np.log10(Mg), gamma), ls='--', lw=1, label=label)

  ax.axvspan(EDGE_r.min(), EDGE_r[fit_range_arr].min(), facecolor='k', alpha=0.1)
  ax.axvspan(EDGE_r[fit_range_arr].max(), EDGE_r.max(), facecolor='k', alpha=0.1)

  ax.set_xlim(*EDGE_r[[0,-1]])

  ax.set_ylabel(r'$\rho_\mathrm{tot}$ [M$_{\odot}$ kpc$^{-3}$]', fontsize=fs)
  ax.set_xlabel('Radius [kpc]', fontsize=fs)

  ax.tick_params(axis='both', which='both', labelsize=fs-2)

  ax.legend(loc='lower left', fontsize=fs-4)
#--------------------------------------------------------------------------

# Trace the orbit backwards until the time of birth:
#--------------------------------------------------------------------------
import agama

# Set up units:
agama.setUnits(mass=1, length=1, velocity=1)
timeunit = pynbody.units.s * pynbody.units.kpc / pynbody.units.km
Gyr_to_timeunit = pynbody.array.SimArray(1, units='Gyr').in_units(timeunit)

# Spherically symmetric Dehnen potential that matches the fit:
Dehnen_potential = agama.Potential(type='Dehnen', mass=Mg, scaleRadius=rs, gamma=gamma)

v_circ = func.Dehnen_vcirc(r=np.linalg.norm(GC_pos/1e3), rs=rs, Mg=Mg, gamma=gamma, G=agama.G)
v_circ_astro = pynbody.array.SimArray(v_circ, units='km s^-1').in_units('kpc Gyr^-1')
orbital_distance = 2 * np.pi * np.linalg.norm(GC_pos/1e3)
period = orbital_distance / v_circ_astro

# Calculate orbits:
total_time = 1 / Gyr_to_timeunit # Gyr
time_steps = 1000 # Myr precision
phase = np.append(GC_pos/1e3, GC_vel * -1)
orbits = agama.orbit(ic=phase, potential=Dehnen_potential, time=total_time, trajsize=time_steps)

# Retrieve the position and velocity at the time of birth:
birthtime = (GC_time - GC_birthtime/1e3) / Gyr_to_timeunit # Gyr
birthindex = np.abs(orbits[0] - birthtime).argmin()
GC_pos_birth = orbits[1][birthindex,[0,1,2]]
GC_vel_birth = orbits[1][birthindex,[3,4,5]] * -1

# Find orbital peri and apo-centre:
Rperi, Rapo = Dehnen_potential.Rperiapo(phase)

if plot_fit:
  ax.text(Rperi, 0.99, r'$R_{\rm peri}$', fontsize=fs-2, color='r', rotation=90, ha='right', va='top', transform=ax.get_xaxis_transform())
  ax.text(Rapo, 0.99, r'$R_{\rm apo}$', fontsize=fs-2, color='r', rotation=90, ha='left', va='top', transform=ax.get_xaxis_transform())
  ax.axvline(Rperi, c='r', lw=0.5, zorder=-100)
  ax.axvline(Rapo, c='r', lw=0.5, zorder=-100)
  ax.axvspan(Rperi, Rapo, facecolor='r', alpha=0.1, zorder=-100)
  ax.axvline(np.linalg.norm(GC_pos_birth), c='r', lw=1, zorder=100)

print('>    Integrated orbit backwards by %.2f Gyr with Agama.' % birthtime)
#--------------------------------------------------------------------------

# Calculate Nbody time unit [doesn't match the one in Nbody6 for some reason]:
#--------------------------------------------------------------------------
G = pynbody.units.G.in_units('pc^3 Msol^-1 Myr^-2')
a = GC_hlr / 1.3 # Plummer scale length
E = (-3 * np.pi / 64.) * (G * GC_mass**2)/a # Energy
T_NB = (G * GC_mass**(5/2.)) / (4 * np.abs(E))**(3/2.)
# As taken from Section 12 in https://wwwstaff.ari.uni-heidelberg.de/mitarbeiter/spurzem/lehre/WS17/cuda/nbody6++_manual.pdf

steps_per_Gyr = 250
age_max = 13.8 * 1e3 # [Myr]
steps = int(steps_per_Gyr * age_max / 1e3)
output_f = max(1, int(np.floor(age_max / (T_NB * steps))))
max_output = int(age_max / T_NB)

print()
print('For %i outputs over %.2f Gyr:' % (steps, age_max/1e3))
print('output frequency = %i' % output_f)
print('Max output = %i' % max_output)
print()
#--------------------------------------------------------------------------

# Save the results to a file:
#--------------------------------------------------------------------------
if save_results:
  if not os.path.isdir(path + f'Nbody6_sims/{suite}_files/'):
    os.mkdir(path + f'Nbody6_sims/{suite}_files/')
  with open(path + f'Nbody6_sims/{suite}_files/{EDGE_sim_name}_{EDGE_output}_{internal_ID}.txt', 'w') as file:
    file.write(sim + '\n')
    file.write('%.8f\n' % GC_mass)
    file.write('%.8f\n' % GC_hlr)
    file.write('%.8f\n' % 10**GC_Z)
    file.write('%.8f\n' % (GC_birthtime/1e3))
    file.write('%.8f\n' % binary_fraction)
    file.write('0.0 0.0 0.0 0.0 0.0 0.0 %.6f \n' % Mg)
    file.write('%.6f %.6f %.6f %.6f %.6f %.6f\n' % (GC_pos_birth[0], GC_pos_birth[1], GC_pos_birth[2], \
                                                    GC_vel_birth[0], GC_vel_birth[1], GC_vel_birth[2]))

  print('>    Parameter file saved to ' + path + f'Nbody6_sims/{suite}_files/{EDGE_sim_name}_{EDGE_output}_{internal_ID}.txt')
else:
  print('>    Parameter file not saved.')
#--------------------------------------------------------------------------
