from config import *

import numpy as np
import pynbody
import tangos
import GC_functions as func
import sys

from lmfit import Parameters, Model

import default_setup
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt, matplotlib.patches as patches
plt.ion()

# Enter EDGE1 to terminal to load pynbody modules

# Simulation choices:
#--------------------------------------------------------------------------
profile_types = ('DM', 'Full', 'fantasy_core')
profile_type = profile_types[0]
binary_fractions = (0, 0.95)
binary_fraction = binary_fractions[0]
#--------------------------------------------------------------------------

# Parameters:
#--------------------------------------------------------------------------
GC_ID = int(sys.argv[1])
plot_fit = True
save_results = True
#--------------------------------------------------------------------------

# Load the GC properties from file:
#--------------------------------------------------------------------------
data = np.genfromtxt('./files/GC_property_table.txt', unpack=True, skip_header=2, dtype=None)[GC_ID-1]

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
#--------------------------------------------------------------------------

# Load the simulation snapshot:
#--------------------------------------------------------------------------
tangos.core.init_db(TANGOS_path + EDGE_sim_name.split('_')[0] + '.db')
session = tangos.core.get_default_session()

h = tangos.get_halo(('/').join([EDGE_sim_name, EDGE_output, 'halo_%i' % EDGE_halo]))
if np.linalg.norm(GC_pos)/1e3 > 40.:
  print('The GC position is beyond the R200 radius. Do not simulate.')
  sys.exit(0)

# Get EDGE simulation density, averaged over next 3 simulation snapshots:
r_range = (0.02, 10)
fit_range = (0.035, 3)
if 'CHIMERA' in EDGE_sim_name:
  data1 = np.loadtxt('./files/CHIMERA_massive/rho_%s.txt' % h.previous.timestep.extension, unpack=True)
  data2 = np.loadtxt('./files/CHIMERA_massive/rho_%s.txt' % h.timestep.extension, unpack=True)
  data3 = np.loadtxt('./files/CHIMERA_massive/rho_%s.txt' % h.next.timestep.extension, unpack=True)
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

# Scale the mass according to the stellar mass evolution and
# the time between GC birth and EDGE snapshot:
######### Need to rethink this element too! #########
#--------------------------------------------------------------------------
#'''
time_difference = (h.calculate('t()')*1e3) - GC_birthtime
count_IDs, mass_mults, hmr_mults, _ = np.loadtxt('./files/GC_multipliers_CHIMERA_massive.txt', unpack=True)
mass_mult = mass_mults[count_IDs == count_ID]
hmr_mult = hmr_mults[count_IDs == count_ID]
GC_mass *= mass_mult
GC_hlr *= min(max(hmr_mult, 0.7), 1.0)
#'''

import stellar_devolution_functions as StellarDevolution

param = StellarEvolution.Parameters('EDGE1')
stars = StellarDevolution.Stars()

for i in stellar_particles:
  stars.add_stars(mass_in_Msol, metal_in_dex, age_in_Myr)

# New stars, evolve independently!
# Perform this a large number of times and take the medians?
for j in range(time_difference_in_Myr):
  stars.evolve(dt, param)

# Add up masses to find initial mass:
GC_mass = np.sum(stars.mass)
#--------------------------------------------------------------------------

# Reconstruct the potential at this time:
#--------------------------------------------------------------------------
def interp(param, alpha):
  return alpha*param[0] + (1-alpha)*param[1]

import pickle
filename = './files/host_profiles_dict.pk1'
with open(filename, 'rb') as file:
  props = pickle.load(file)

sim = '_'.join([EDGE_sim_name, profile_type])
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

  ax.legend(fontsize=fs-4)
#--------------------------------------------------------------------------

# Trace the orbit backwards until the time of birth:
#--------------------------------------------------------------------------
GC_birthtime = h.calculate('t()') - 0.1
GC_time = h.calculate('t()')

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
birthtime = (GC_time - GC_birthtime) / Gyr_to_timeunit # Gyr
birthindex = np.abs(orbits[0] - birthtime).argmin()
GC_pos_birth = orbits[1][birthindex,[0,1,2]]
GC_vel_birth = orbits[1][birthindex,[3,4,5]] * -1

# Find orbital peri and apo-centre:
Rperi, Rapo = Dehnen_potential.Rperiapo(phase)

ax.text(Rperi, 0.99, r'$R_{\rm peri}$', fontsize=fs-2, color='r', rotation=90, ha='right', va='top', transform=ax.get_xaxis_transform())
ax.text(Rapo, 0.99, r'$R_{\rm apo}$', fontsize=fs-2, color='r', rotation=90, ha='left', va='top', transform=ax.get_xaxis_transform())
ax.axvline(Rperi, c='r', lw=0.5, zorder=-100)
ax.axvline(Rapo, c='r', lw=0.5, zorder=-100)
ax.axvspan(Rperi, Rapo, facecolor='r', alpha=0.1, zorder=-100)

print('>    Integrated orbit backwards by %.2f Gyr with Agama.' % birthtime)
#--------------------------------------------------------------------------

# Calculate Nbody time unit:
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

# Save the results to a file:
#--------------------------------------------------------------------------
if save_results:
  with open(path + f'/files/{EDGE_sim_name}_{EDGE_output}_{count_ID}.txt', 'w') as file:
    file.write(sim + '\n')
    file.write('%.8f\n' % GC_mass)
    file.write('%.8f\n' % GC_hlr)
    file.write('%.8f\n' % 10**GC_Z)
    file.write('%.8f\n' % GC_birthtime)
    file.write('%.8f\n' % binary_fraction)
    file.write('0.0 0.0 0.0 0.0 0.0 0.0 %.6f \n' % Mg)
    file.write('%.6f %.6f %.6f %.6f %.6f %.6f\n' % (GC_pos_birth[0], GC_pos_birth[1], GC_pos_birth[2], \
                                                  GC_vel_birth[0], GC_vel_birth[1], GC_vel_birth[2]))

  print('>    Parameter file saved to ' + f'/files/{EDGE_sim_name}_{EDGE_output}_{count_ID}.txt')
else:
  print('>    Parameter file not saved.')
#--------------------------------------------------------------------------
