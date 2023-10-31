from config import *

import numpy as np
import pynbody
import tangos
import GC_functions as func
import sys

from lmfit import Parameters, Model

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt, matplotlib.patches as patches
plt.ioff()

# Enter EDGE1 to terminal to load pynbody modules

GC_ID = int(sys.argv[1])
plot_fit = True
save_results = True

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

if GC_hlr <= 15:
  print('This GC is too diffuse.')
  sys.exit(0)
#--------------------------------------------------------------------------

# Load the simulation snapshot:
#--------------------------------------------------------------------------
tangos.core.init_db(TANGOS_path + EDGE_sim_name.split('_')[0] + '.db')
session = tangos.core.get_default_session()

h = tangos.get_halo(('/').join([EDGE_sim_name, EDGE_output, 'halo_%i' % EDGE_halo]))
if np.linalg.norm(GC_pos)/1e3 > h['r200c']*1.1:
  print('The GC position is beyond the R200 radius. Do not simulate.')
  sys.exit(0)

# Get EDGE simulation density, averaged over next 3 simulation snapshots:
nmax = 3
EDGE_rho, EDGE_r = h.previous.calculate_for_descendants('dm_density_profile', 'rbins_profile', nmax=nmax)
for matter in ['gas', 'stars']:
  if f'{matter}_density_profile' in h.keys():
    EDGE_rho += h.previous.calculate_for_descendants(f'{matter}_density_profile', nmax=nmax)[0]

# Rebin the density:
N_bins = 200
r_min = 0.03
r_max = 3
fit_min = 0.1
fit_max = 1

if np.linalg.norm(GC_pos)/1e3 > fit_max:
  print('GC position is greater that fit_max, %.2f kpc.' % fit_max)
elif np.linalg.norm(GC_pos)/1e3 < fit_min:
  print('GC position is less that fit_min, %.2f kpc.' % fit_min)

new_r = np.logspace(np.log10(r_min), np.log10(r_max), N_bins)

for i in range(nmax + 1):
  EDGE_r[i], EDGE_rho[i] = func.rebin(EDGE_r[i], EDGE_rho[i], new_r)
EDGE_rho = np.median(list(EDGE_rho), axis=0)
EDGE_r = EDGE_r[0]
fit_range = (EDGE_r > fit_min) & (EDGE_r <= fit_max)

print(f'>    Loaded EDGE density profile for {EDGE_sim_name}, {EDGE_output}.')
#--------------------------------------------------------------------------

# Scale the mass according to the stellar mass evolution and
# the time between GC birth and EDGE snapshot:
#--------------------------------------------------------------------------
'''
time_difference = (h.calculate('t()')*1e3) - GC_birthtime
count_IDs, mass_mults, hmr_mults, _ = np.loadtxt('./files/GC_multipliers.txt', unpack=True)
mass_mult = mass_mults[count_IDs == count_ID]
hmr_mult = hmr_mults[count_IDs == count_ID]
GC_mass *= mass_mult
GC_hlr *= min(max(hmr_mult, 0.7), 1.0)
'''
#--------------------------------------------------------------------------

# Reconstruct the potential at this time:
#--------------------------------------------------------------------------
profile_data = np.genfromtxt('./files/EDGE_profiles.txt', unpack=True, dtype=None)
row = np.where([EDGE_sim_name.split('_')[0] in i[0].decode('ascii') for i in profile_data])[0][0]
Mg = profile_data[row][1]
rs_epoch = [profile_data[row][i] for i in [2,3,4,5,6,7]]
gamma_epoch = [profile_data[row][i] for i in [8,9,10,11,12,13]]

def interp(param, alpha):
  return alpha*param[0] + (1-alpha)*param[1]

if (GC_birthtime < 1000):
  ALP = (GC_birthtime-0.) / (1000.-0.)
  rs = ALP*rs_epoch[1] + (1.0-ALP)*rs_epoch[0]
  gamma = ALP*gamma_epoch[1] + (1.0-ALP)*gamma_epoch[0]
elif (GC_birthtime < 2000) & (GC_birthtime > 1000):
  ALP = (GC_birthtime-1000.) / (2000.-1000.)
  rs = ALP*rs_epoch[2] + (1.0-ALP)*rs_epoch[1]
  gamma = ALP*gamma_epoch[2] + (1.0-ALP)*gamma_epoch[1]
elif (GC_birthtime < 4000) & (GC_birthtime < 2000):
  ALP = (GC_birthtime-2000.) / (4000.-2000.)
  rs = ALP*rs_epoch[3] + (1.0-ALP)*rs_epoch[2]
  gamma = ALP*gamma_epoch[3] + (1.0-ALP)*gamma_epoch[2]
elif (GC_birthtime < 6000) & (GC_birthtime > 4000):
  ALP = (GC_birthtime-4000.) / (6000.-4000.)
  rs = ALP*rs_epoch[4] + (1.0-ALP)*rs_epoch[3]
  gamma = ALP*gamma_epoch[4] + (1.0-ALP)*gamma_epoch[3]
elif (GC_birthtime < 14000):
  ALP = (GC_birthtime-6000.) / (13800.-6000.)
  rs = ALP*rs_epoch[5] + (1.0-ALP)*rs_epoch[4]
  gamma = ALP*gamma_epoch[5] + (1.0-ALP)*gamma_epoch[4]
#--------------------------------------------------------------------------

# Plot the result and compare:
#--------------------------------------------------------------------------
if plot_fit:
  fs = 14
  fig, ax = plt.subplots(figsize=(6, 6))

  ax.loglog(EDGE_r, EDGE_rho, 'k', lw=2, label='%s, %s' % (EDGE_sim_name, EDGE_output))
  ax.axvline(np.linalg.norm(GC_pos/1e3), c='k', lw=1)

  label = r'Fit %i: ' % i + \
          r'$r_{\rm s}=%.2f$, ' % (rs) + \
          r'$M_{\rm g}=%s\,$M$_{\odot}$, ' % func.latex_float(Mg) + \
          r'$\gamma=%.2g$' % gamma
  ax.loglog(EDGE_r, func.Dehnen_profile(EDGE_r, np.log10(rs), np.log10(Mg), gamma), ls='--', lw=1, label=label)

  ax.axvspan(EDGE_r.min(), EDGE_r[fit_range].min(), facecolor='k', alpha=0.1)
  ax.axvspan(EDGE_r[fit_range].max(), EDGE_r.max(), facecolor='k', alpha=0.1)

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
    file.write('%.8f\n' % GC_mass)
    file.write('%.8f\n' % GC_hlr)
    file.write('%.8f\n' % 10**GC_Z)
    file.write('%.8f\n' % GC_birthtime)
    file.write('0.0 0.0 0.0 0.0 0.0 0.0 %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.2f %.2f %.2f %.2f %.2f %.2f\n' \
               % (Mg, rs_epoch[0], rs_epoch[1], rs_epoch[2], rs_epoch[3], rs_epoch[4], rs_epoch[5], \
                  gamma_epoch[0], gamma_epoch[1], gamma_epoch[2], gamma_epoch[3], gamma_epoch[4], gamma_epoch[5]))
    file.write('%.6f %.6f %.6f %.6f %.6f %.6f\n' % (GC_pos_birth[0], GC_pos_birth[1], GC_pos_birth[2], \
                                                  GC_vel_birth[0], GC_vel_birth[1], GC_vel_birth[2]))

  print('>    Parameter file saved to ' + f'/files/{EDGE_sim_name}_{EDGE_output}_{count_ID}.txt')
else:
  print('>    Parameter file not saved.')
#--------------------------------------------------------------------------
