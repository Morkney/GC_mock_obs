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
# Halo1445 192474380.36046273 1.4884956245020209 0.9286634547921888 1.1030368592595123 0.8245889922520259 0.8458253028330703 0.8410221710629264 1.0 0.75 1.0 0.75 0.75 0.75
# Halo1459 187869975.68623495 0.8996810852229837 0.6213171830221742 0.6840549249545358 0.6997792085977168 0.6990678013797488 0.6957148245620229 1.0 0.75 0.75 0.75 0.75 0.75
# Halo600  714819341.9016784 3.1229024619035886 2.8584164179921077 2.4640192326470753 2.238263632239617 2.0150158546651804 2.04436606464254 1.0 1.0 1.0 1.0 1.0 1.0
# Halo605  1300407253.7867322 3.513737117188465 1.3368461549689457 2.370820995584988 2.3835600607216647 2.314759972099984 2.2847718128111487 1.0 0.5 1.0 1.0 1.0 1.0
# Halo624  691714733.1018294 2.187650824075687 1.8832524040766991 1.9618154180487481 1.331853101334287 1.3123061886661298 1.6899368375197799 1.0 1.0 1.0 0.75 0.75 1.0
#--------------------------------------------------------------------------

# Load the simulation database:
#--------------------------------------------------------------------------
EDGE_sim_name = 'Halo605_fiducial_hires'
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
# - Find the nearest gamma in multiples of 1/4.
# - Refit with this gamma.
# - Additionally, mass must be conserved. Keep the mass from the very first fit!
#--------------------------------------------------------------------------
params = ['log_rs', 'log_Mg', 'gamma']
priors = {}
for param in params:
  priors[param] = {}

priors['log_rs']['guess'] = np.log10(0.1)
priors['log_rs']['min'] = -3
priors['log_rs']['max'] = 2

priors['log_Mg']['guess'] = np.log10(h['M200c'])
priors['log_Mg']['min'] = 7.5
priors['log_Mg']['max'] = 10

priors['gamma']['guess'] = 0.5
priors['gamma']['min'] = 0.0
priors['gamma']['max'] = 1.0

vary=False

# Add these constraints to the Dehnen model:
Dehnen = Model(func.Dehnen_profile)
fit_params = Parameters()
for param in params:
  fit_params.add(param, value=priors[param]['guess'], min=priors[param]['min'], max=priors[param]['max'])
fixed_fit_params = fit_params.copy()
weights = np.sqrt(np.arange(len(EDGE_r[0][fit_range])))

# Loop over each step and perform the fit:
gammas = []
rs = []
for i in range(len(EDGE_t)):
  print('>    %i' % i)
  # First pass to find best gamma:
  fit = Dehnen.fit(EDGE_rho[i][fit_range], fit_params, r=EDGE_r[i][fit_range], weights=weights)
  fixed_gamma = round(fit.best_values['gamma']*4)/4.
  fixed_fit_params.add('gamma', value=fixed_gamma, vary=False)
  # Second pass to find fit using best gamma:
  fit = Dehnen.fit(EDGE_rho[i][fit_range], fixed_fit_params, r=EDGE_r[i][fit_range], weights=weights)
  # Lock down Mg if this is the z=0 step:
  if i==0:
    fit_params.add('log_Mg', value=fit.best_values['log_Mg'], vary=False)
    fixed_fit_params.add('log_Mg', value=fit.best_values['log_Mg'], vary=False)
  gammas.append(fit.best_values['gamma'])
  rs.append(10**fit.best_values['log_rs'])
gammas = np.array(gammas)
rs = np.array(rs)
Mg = 10**fit.best_values['log_Mg']

# Check a fit:
i = 0
fs = 14
fig, ax = plt.subplots(figsize=(6, 6))
ax.loglog(EDGE_r[i], EDGE_rho[i], 'k', lw=2)
ax.loglog(EDGE_r[i], func.Dehnen_profile(EDGE_r[i], np.log10(rs[i]), np.log10(Mg), gammas[i]), ls='--', lw=1)

# Plot the evolution of the fit parameters:
fs = 14
fig, ax = plt.subplots(figsize=(6, 6))
#ax.plot(EDGE_t[:len(gammas)], gammas)
ax.plot(EDGE_t[:len(gammas)], rs)

# Find the most reprentative profile out of five time epochs:
# 1, 2, 4, 6, 13.8
t_epochs = [(0.3, 0.7), (0.8, 1.2), (1.5, 2.5), (3, 5), (5, 7), (12, 14)]
gamma_epoch = np.empty(6)
rs_epoch = np.empty(6)
for i in range(6):
  t_epoch = np.where((EDGE_t>=t_epochs[i][0]) & (EDGE_t<=t_epochs[i][1]))[0]
  gamma_epoch[i] = round(np.median(gammas[t_epoch])*4)/4.
  rs_epoch[i] = np.median(rs[t_epoch][gammas[t_epoch]==gamma_epoch[i]])

errorbars = np.abs(np.array(t_epochs).T - np.mean([i for i in t_epochs], axis=1))
ax.errorbar(np.mean([i for i in t_epochs], axis=1), rs_epoch, xerr=errorbars, ecolor='k', lw=0, elinewidth=2, zorder=100)
ax.set_yscale('log')

# Plot goodness of fit:
def interp(param, alpha):
  return alpha*param[0] + (1-alpha)*param[1]

def make_plot(i):
  t = EDGE_t[i]
  if (t < 1):
    profile0 = func.Dehnen_profile(EDGE_r[i], np.log10(rs_epoch[0]), np.log10(Mg), gamma_epoch[0])
    profile1 = func.Dehnen_profile(EDGE_r[i], np.log10(rs_epoch[1]), np.log10(Mg), gamma_epoch[1])
    profile = interp([profile1, profile0], (t) / (1.-0.))
  elif (t < 2) & (t > 1):
    profile0 = func.Dehnen_profile(EDGE_r[i], np.log10(rs_epoch[1]), np.log10(Mg), gamma_epoch[1])
    profile1 = func.Dehnen_profile(EDGE_r[i], np.log10(rs_epoch[2]), np.log10(Mg), gamma_epoch[2])
    profile = interp([profile1, profile0], (t-1.) / (2.-1.))
  elif (t < 4) & (t < 2):
    profile0 = func.Dehnen_profile(EDGE_r[i], np.log10(rs_epoch[2]), np.log10(Mg), gamma_epoch[2])
    profile1 = func.Dehnen_profile(EDGE_r[i], np.log10(rs_epoch[3]), np.log10(Mg), gamma_epoch[3])
    profile = interp([profile1, profile0], (t-2.) / (4.-2.))
  elif (t < 6) & (t > 4):
    profile0 = func.Dehnen_profile(EDGE_r[i], np.log10(rs_epoch[3]), np.log10(Mg), gamma_epoch[3])
    profile1 = func.Dehnen_profile(EDGE_r[i], np.log10(rs_epoch[4]), np.log10(Mg), gamma_epoch[4])
    profile = interp([profile1, profile0], (t-4.) / (6.-4.))
  elif (t < 14):
    profile0 = func.Dehnen_profile(EDGE_r[i], np.log10(rs_epoch[4]), np.log10(Mg), gamma_epoch[4])
    profile1 = func.Dehnen_profile(EDGE_r[i], np.log10(rs_epoch[5]), np.log10(Mg), gamma_epoch[5])
    profile = interp([profile1, profile0], (t-6.) / (13.8-6.))
  fs = 14
  fig, ax = plt.subplots(figsize=(6, 6))
  ax.loglog(EDGE_r[i], EDGE_rho[i], 'k', lw=2)
  ax.loglog(EDGE_r[i], profile, ls='--', lw=1)

print(Mg, ' '.join(['%s' % i for i in rs_epoch]), ' '.join(['%s' % i for i in gamma_epoch]))
