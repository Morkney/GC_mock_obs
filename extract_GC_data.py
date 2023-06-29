from config import *

import numpy as np
import pynbody
import tangos

from lmfit import Parameters, Model

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt, matplotlib.patches as patches
plt.ion()

plot_fit = True
save_results = True

# Load the simulation snapshot:
#--------------------------------------------------------------------------
h = tangos.get_halo(('/').join([EDGE_sim_name, EDGE_output, 'halo_%i' % EDGE_halo]))
#--------------------------------------------------------------------------

# Fit a Dehnen potential density profile to the simulation density profile:
#--------------------------------------------------------------------------
# Reformat scientific notation:
def latex_float(f):
    float_str = "{0:.1e}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

def Dehnen_profile(r, log_rs, log_Mg, gamma):
  rs = 10**log_rs
  Mg = 10**log_Mg
  rho_0 = Mg * (3. - gamma) / (4. * np.pi)
  return rho_0 * rs / ((r**gamma) * (r + rs)**(4 - gamma))

# Define the circular velocity in a Dehnen profile [Dehnen 1993]:
def v_circ_dehnen(r, rs, Mg, gamma, G):
  return np.sqrt(G * Mg * r**(2 - gamma) / (r + rs)**(3 - gamma))

def rebin(x, y, new_x_bins, operation='mean'):

  index_radii = np.digitize(x, new_x_bins)
  if operation is 'mean':
    new_y = np.array([y[index_radii == i].mean() for i in range(1, len(new_x_bins))])
  elif operation is 'sum':
    new_y = np.array([y[index_radii == i].sum() for i in range(1, len(new_x_bins))])
  else:
    raise ValueError("Operation not supported")

  new_x = (new_x_bins[1:] + new_x_bins[:-1]) / 2
  new_y[np.isnan(new_y)] = np.finfo(float).eps

  return new_x, new_y

# Get EDGE simulation density:
# Average over the next 3 snapshots:
nmax = 3
EDGE_rho, EDGE_r = h.calculate_for_descendants('dm_density_profile', 'rbins_profile', nmax=nmax)
for matter in ['gas', 'stars']:
  if f'{matter}_density_profile' in h.keys():
    EDGE_rho += h.calculate_for_descendants(f'{matter}_density_profile', nmax=nmax)[0]

# Reduce number of bins:
N_bins = 200
r_min = 0.03
r_max = 3
fit_min = 0.1
fit_max = 1
new_r = np.logspace(np.log10(r_min), np.log10(r_max), N_bins)

for i in range(nmax+1):
  EDGE_r[i], EDGE_rho[i] = rebin(EDGE_r[i], EDGE_rho[i], new_r)
EDGE_rho = np.median(list(EDGE_rho), axis=0)
EDGE_r = np.mean(EDGE_r)
fit_range = (EDGE_r > fit_min) & (EDGE_r <= fit_max)
#--------------------------------------------------------------------------

# Prepare to make a profile fit:
#--------------------------------------------------------------------------

# Define the initial parameter guesses and constraints:
params = ['rs', 'Mg', 'gamma']
priors = {}
for param in params:
  priors[param] = {}

priors['rs']['guess'] = np.log10(0.1)
priors['rs']['min'] = -3
priors['rs']['max'] = 2

priors['Mg']['guess'] = np.log10(h['M200c'])
priors['Mg']['min'] = 7.5
priors['Mg']['max'] = 10

# Gamma in Nbody6df is limited to multiples of 1/4 for technical reasons.
# Discrete values cannot be used in lmfit, so we need to create a fit with each gamma value in turn:
gammas = [0, 0.25, 0.5, 0.75, 1.0]
vary = False

# Add these constraints to the Dehnen model:
Dehnen = Model(Dehnen_profile)
params = Parameters()

params.add('log_rs', value=priors['rs']['guess'], min=priors['rs']['min'], max=priors['rs']['max'])
params.add('log_Mg', value=priors['Mg']['guess'], min=priors['Mg']['min'], max=priors['Mg']['max'])

fits = []
for gamma in gammas:

  print('Fitting for gamma=%.2g' % gamma)

  params.add('gamma', value=gamma, vary=vary)

  # Perform the fit:
  fits.append(Dehnen.fit(EDGE_rho[fit_range], params, r=EDGE_r[fit_range]))

# Plot the result and compare:
if plot_fit:
  fs = 10
  fig, ax = plt.subplots(figsize=(6, 6))

  ax.loglog(EDGE_r, EDGE_rho, 'k', lw=2, label='%s, %s' % (EDGE_sim_name, EDGE_output))
  ax.axvline(np.linalg.norm(GC_pos/1e3), c='k', lw=1)

  lines = [''] * len(fits)

  for i, fit in enumerate(fits):
    label = r'Fit %i: ' % i + \
            r'$r_{\rm s}=%.2f$, ' % (10**fit.best_values['log_rs']) + \
            r'$M_{\rm g}=%s\,$M$_{\odot}$, ' % latex_float(10**fit.best_values['log_Mg']) + \
            r'$\gamma=%.2g$' % fit.best_values['gamma']
    lines[i], = ax.loglog(EDGE_r, Dehnen_profile(EDGE_r, **fit.best_values), ls='--', lw=1, label=label)

  ax.axvspan(EDGE_r.min(), EDGE_r[fit_range].min(), facecolor='k', alpha=0.1)
  ax.axvspan(EDGE_r[fit_range].max(), EDGE_r.max(), facecolor='k', alpha=0.1)

  ax.set_xlim(*EDGE_r[[0,-1]])

  ax.set_ylabel(r'$\rho_\mathrm{tot}$ [M$_{\odot}$ kpc$^{-3}$]')
  ax.set_xlabel('Radius [kpc]')

  ax.legend(fontsize=fs-2)
#--------------------------------------------------------------------------

# Select preferred fit:
#--------------------------------------------------------------------------
try:
  choice = int(input('Enter your preferred fit: '))
except ValueError:
  print('Enter an integer 0-4.')
fit = fits[choice]

lines[choice].set_linewidth(2)
ax.legend(fontsize=fs-2)
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
mass = 10**fit.best_values['log_Mg']
rs = 10**fit.best_values['log_rs']
gamma = fit.best_values['gamma']
Dehnen_potential = agama.Potential(type='Dehnen', mass=mass, scaleRadius=rs, gamma=gamma)

v_circ = v_circ_dehnen(r=np.linalg.norm(GC_pos/1e3), rs=rs, Mg=mass, gamma=gamma, G=agama.G)
v_circ_astro = pynbody.array.SimArray(v_circ, units='km s^-1').in_units('kpc Gyr^-1')
orbital_distance = 2 * np.pi * np.linalg.norm(GC_pos/1e3)
period = orbital_distance / v_circ_astro
#GC_vel = np.array([0, v_circ, 0])
#print('Velocity / circular velocity = %.2f' % (np.linalg.norm(GC_vel) / v_circ))

# Calculate orbits:
total_time = 1 / Gyr_to_timeunit # Gyr
time_steps = 1000 # Myr precision
phase = np.append(GC_pos/1e3, GC_vel * -1)
orbits = agama.orbit(ic=phase, potential=Dehnen_potential, time=total_time, trajsize=time_steps)

# Retrieve the position and velocity at the time of birth:
birthtime = (GC_time - GC_birthtime) / Gyr_to_timeunit # Gyr
birthindex = np.abs(orbits[0] - birthtime).argmin()
GC_pos_birth = orbits[1][birthindex,[0,1,2]] * 1e3 # Convert to pc
GC_vel_birth = orbits[1][birthindex,[3,4,5]] * -1

# Find orbital peri and apo-centre:
Rperi, Rapo = Dehnen_potential.Rperiapo(phase)

ax.text(Rperi, 0.99, r'$R_{\rm peri}$', fontsize=fs-2, color='r', rotation=90, ha='right', va='top', transform=ax.get_xaxis_transform())
ax.text(Rapo, 0.99, r'$R_{\rm apo}$', fontsize=fs-2, color='r', rotation=90, ha='left', va='top', transform=ax.get_xaxis_transform())
ax.axvline(Rperi, c='r', lw=0.5, zorder=-100)
ax.axvline(Rapo, c='r', lw=0.5, zorder=-100)
ax.axvspan(Rperi, Rapo, facecolor='r', alpha=0.1, zorder=-100)
#--------------------------------------------------------------------------

# Save the results to a file:
#--------------------------------------------------------------------------
if save_results:
  with open(path + f'/files/{EDGE_sim_name}_{EDGE_output}_{sim_name}.txt', 'w') as file:
    file.write('%.3f\n' % GC_mass)
    file.write('%.3f\n' % GC_hlr)
    file.write('%.3f\n' % 10**GC_Z)
    file.write('0.0 0.0 0.0 0.0 0.0 0.0 %.6f %.6f %.2f\n' % (mass, rs, gamma))
    file.write('%.6f %.6f %.6f %.6f %.6f %.6f\n' % (GC_pos_birth[0], GC_pos_birth[1], GC_pos_birth[2], \
                                                  GC_vel_birth[0], GC_vel_birth[1], GC_vel_birth[2]))
#--------------------------------------------------------------------------
