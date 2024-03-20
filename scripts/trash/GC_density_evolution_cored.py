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
plt.ion()

# The fits:
#--------------------------------------------------------------------------
# Halo1445: 104746313.6830042 0.38537074877914856 0.3560388418459976 0.35644320480382335 0.3571910836597091 0.36751249260269203 0.37486614156289155 0 0 0 0 0 0
# Halo1459: 132605644.68581183 0.6002343182428231 0.3012749659999634 0.3472379683852327 0.3582053568100258 0.3619482284474273 0.36087173496590347 0 0 0 0 0 0
# Halo600: 342157900.17760575 0.6522108972168827 0.7545268000017404 0.87004181916364 0.7855637289906934 0.6629070859545099 0.6470623684954403 0 0 0 0 0 0
# Halo605: 308092584.3570545 0.3223964826125079 0.506939088695306 0.4875598568960232 0.4905460335992703 0.4847884334482364 0.487419878233026 0 0 0 0 0 0
# Halo624: 246687609.61447474 0.23022770488322677 0.4541200228597608 0.4891151276476255 0.46668789050529585 0.46033606627369383 0.44548037471406515 0 0 0 0 0 0
#--------------------------------------------------------------------------

# Load the simulation database:
#--------------------------------------------------------------------------
EDGE_sim_name = 'Halo600_fiducial_hires'
tangos.core.init_db(TANGOS_path + EDGE_sim_name.split('_')[0] + '.db')
session = tangos.core.get_default_session()
#--------------------------------------------------------------------------

# Find the latest halo:
#--------------------------------------------------------------------------
output = tangos.get_simulation(EDGE_sim_name).timesteps[-1].extension
h = tangos.get_halo(EDGE_sim_name + '/' + output + '/' + 'halo_1')
#--------------------------------------------------------------------------

# Find the density profile evolution with time:
#--------------------------------------------------------------------------
EDGE_t, EDGE_rho, EDGE_r = h.calculate_for_progenitors('t()', 'dm_density_profile+gas_density_profile+star_density_profile', 'rbins_profile')

# Rebin the density:
N_bins = 100+1
r_min = 0.02
r_max = 3
fit_min = 0.03
fit_max = 2
new_r = np.logspace(np.log10(r_min), np.log10(r_max), N_bins)
for i in range(len(EDGE_t)):
  EDGE_r[i], EDGE_rho[i] = func.rebin(EDGE_r[i], EDGE_rho[i], new_r)
fit_range = (EDGE_r[0] > fit_min) & (EDGE_r[0] <= fit_max)

# Fit a profile to every single timestep.
# - Fit a raw profile.
# - Additionally, mass must be conserved. Keep the mass from the very first fit!
#--------------------------------------------------------------------------
params = ['log_rs', 'log_Mg', 'gamma']
priors = {}
for param in params:
  priors[param] = {}

priors['log_rs']['guess'] = np.log10(0.4)
priors['log_rs']['min'] = np.log10(0.2)
priors['log_rs']['max'] = 2

priors['log_Mg']['guess'] = np.log10(h['M200c'])
priors['log_Mg']['min'] = 7.5
priors['log_Mg']['max'] = 10

priors['gamma']['guess'] = 0.0
priors['gamma']['min'] = None
priors['gamma']['max'] = None

varys=[True, True, False]

# Add these constraints to the Dehnen model:
Dehnen = Model(func.Dehnen_profile)
fit_params = Parameters()
for param, vary in zip(params, varys):
  fit_params.add(param, value=priors[param]['guess'], min=priors[param]['min'], max=priors[param]['max'], vary=vary)
fixed_fit_params = fit_params.copy()

# Loop over each step and perform the fit:
rs = []
for i in range(len(EDGE_t)):
  print('>    %i' % i)

  if 'stellar_3D_halflight' in h.keys():
    fit_min = max(np.min(h.calculate_for_progenitors('stellar_3D_halflight', nmax=3)), 0.03)
  else:
    fit_min = 0.03
  fit_range = (EDGE_r[0] > fit_min) & (EDGE_r[0] <= fit_max)
  if np.sum(fit_range) < 1:
    rs.append(10**fit.best_values['log_rs'])
    h = h.previous
    continue

  weights = np.sqrt(np.arange(len(EDGE_r[0][fit_range])))
  fit = Dehnen.fit(EDGE_rho[i][fit_range], fit_params, r=EDGE_r[i][fit_range])#, weights=weights)

  # Lock down Mg if this is the z=0 step:
  if i==0:
    fit_params.add('log_Mg', value=fit.best_values['log_Mg'], vary=False)
  rs.append(10**fit.best_values['log_rs'])

  h = h.previous

rs = np.array(rs)
Mg = 10**fit.best_values['log_Mg']

# Check a fit:
i = 0
fs = 14
fig, ax = plt.subplots(figsize=(6, 6))
ax.loglog(EDGE_r[i], EDGE_rho[i], 'k', lw=2)
ax.loglog(EDGE_r[i], func.Dehnen_profile(EDGE_r[i], np.log10(rs[i]), np.log10(Mg), 0), ls='--', lw=1)

# Plot the evolution of the fit parameters:
fs = 14
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(EDGE_t[:len(rs)], rs)

# Find the most reprentative profile out of five time epochs:
# 1, 2, 4, 6, 13.8
t_epochs = [(0.3, 0.7), (0.8, 1.2), (1.5, 2.5), (3, 5), (5, 7), (12, 14)]
rs_epoch = np.empty(6)
for i in range(6):
  t_epoch = np.where((EDGE_t>=t_epochs[i][0]) & (EDGE_t<=t_epochs[i][1]))[0]
  rs_epoch[i] = np.median(rs[t_epoch])

errorbars = np.abs(np.array(t_epochs).T - np.mean([i for i in t_epochs], axis=1))
ax.errorbar(np.mean([i for i in t_epochs], axis=1), rs_epoch, xerr=errorbars, ecolor='k', lw=0, elinewidth=2, zorder=100)
ax.set_yscale('log')

# Plot goodness of fit:
def interp(param, alpha):
  return alpha*param[0] + (1-alpha)*param[1]

def make_plot(i):
  t = EDGE_t[i]
  if (t < 1):
    profile0 = func.Dehnen_profile(EDGE_r[i], np.log10(rs_epoch[0]), np.log10(Mg), 0)
    profile1 = func.Dehnen_profile(EDGE_r[i], np.log10(rs_epoch[1]), np.log10(Mg), 0)
    profile = interp([profile1, profile0], (t) / (1.-0.))
  elif (t < 2) & (t > 1):
    profile0 = func.Dehnen_profile(EDGE_r[i], np.log10(rs_epoch[1]), np.log10(Mg), 0)
    profile1 = func.Dehnen_profile(EDGE_r[i], np.log10(rs_epoch[2]), np.log10(Mg), 0)
    profile = interp([profile1, profile0], (t-1.) / (2.-1.))
  elif (t < 4) & (t < 2):
    profile0 = func.Dehnen_profile(EDGE_r[i], np.log10(rs_epoch[2]), np.log10(Mg), 0)
    profile1 = func.Dehnen_profile(EDGE_r[i], np.log10(rs_epoch[3]), np.log10(Mg), 0)
    profile = interp([profile1, profile0], (t-2.) / (4.-2.))
  elif (t < 6) & (t > 4):
    profile0 = func.Dehnen_profile(EDGE_r[i], np.log10(rs_epoch[3]), np.log10(Mg), 0)
    profile1 = func.Dehnen_profile(EDGE_r[i], np.log10(rs_epoch[4]), np.log10(Mg), 0)
    profile = interp([profile1, profile0], (t-4.) / (6.-4.))
  elif (t < 14):
    profile0 = func.Dehnen_profile(EDGE_r[i], np.log10(rs_epoch[4]), np.log10(Mg), 0)
    profile1 = func.Dehnen_profile(EDGE_r[i], np.log10(rs_epoch[5]), np.log10(Mg), 0)
    profile = interp([profile1, profile0], (t-6.) / (13.8-6.))
  fs = 14
  fig, ax = plt.subplots(figsize=(6, 6))
  ax.loglog(EDGE_r[i], EDGE_rho[i], 'k', lw=2)
  ax.loglog(EDGE_r[i], profile, ls='--', lw=1)

print(Mg, ' '.join(['%s' % i for i in rs_epoch]), ' '.join(['%s' % i for i in [0,0,0,0,0,0]]))
