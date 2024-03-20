from config import *

import numpy as np
import pynbody
pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
import tangos
import GC_functions as func
import sys, os

from lmfit import Parameters, Model

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt, matplotlib.patches as patches
plt.ion()

# The fit:
#--------------------------------------------------------------------------
# Standard 585599232.5297644 1.7594676465660435 1.1867894505482963 0.96389276520189 0.8136608448297856 0.8200190734234991 0.6077890085067078 1.0 0.75 0.5 0.5 0.5 0.0
# Massive 1487558917.3892713 2.96603298297576 2.7625219478307574 1.5076331980250162 2.012022648963931 1.5440669884493339 1.5515509201599285 1.0 1.0 0.75 1.0 0.75 0.75
#--------------------------------------------------------------------------

# Load the simulation database:
#--------------------------------------------------------------------------
EDGE_sim_name = 'Halo383_Massive'
#--------------------------------------------------------------------------

# Find the latest halo:
#--------------------------------------------------------------------------
output = tangos.get_simulation(EDGE_sim_name).timesteps[-1].extension
h = tangos.get_halo(EDGE_sim_name + '/' + output + '/' + 'halo_1')
#--------------------------------------------------------------------------

# Find the density profile evolution with time:
#--------------------------------------------------------------------------
EDGE_t = []
EDGE_r = []
EDGE_rho = []
gammas = []
rs = []

N_bins = 100+1
r_min = 0.03
r_max = 3
fit_min = 0.05
fit_max = 2

params = ['log_rs', 'log_Mg', 'gamma']
priors = {}
for param in params:
  priors[param] = {}

priors['log_rs']['guess'] = np.log10(0.1)
priors['log_rs']['min'] = -3
priors['log_rs']['max'] = 2

priors['log_Mg']['guess'] = 9.5
priors['log_Mg']['min'] = 7.5
priors['log_Mg']['max'] = 11

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

while True:

  print('>    %s' % h.timestep.extension)

  if os.path.isfile('./files/CHIMERA_massive/rho_%s.txt' % h.timestep.extension):
    data = np.loadtxt('./files/CHIMERA_massive/rho_%s.txt' % h.timestep.extension, unpack=True)
    EDGE_r.append(data[0])
    EDGE_rho.append(data[1])
    EDGE_t.append(h.calculate('t()'))
  else:
    break
     # Load the raw simulation data:
    EDGE_t.append(h.calculate('t()'))
    s = pynbody.load('/vol/ph/astro_data/shared/etaylor/CHIMERA/' + EDGE_sim_name + '/' + h.timestep.extension)
    s.physical_units()
    s_h = s.halos()
    s_h = s_h[h.calculate('halo_number()')-1]
    s.g['pos']; s.d['pos']; s.s['pos']
    s.g['mass']; s.d['mass']; s.s['mass']

    # Centre on the main halo:
    #if 'shrink_center' in h.keys():
    #  s['pos'] -= h['shrink_center']
    #else:
    cen = pynbody.analysis.halo.shrink_sphere_center(s_h.d, shrink_factor=0.8)
    s['pos'] -= cen

    # Retrieve the profile:
    DM_prof = pynbody.analysis.profile.Profile(s.d, ndim=3, type='log', min=r_min, max=r_max, nbins=N_bins)
    gas_prof = pynbody.analysis.profile.Profile(s.g, ndim=3, type='log', min=r_min, max=r_max, nbins=N_bins)
    star_prof = pynbody.analysis.profile.Profile(s.s, ndim=3, type='log', min=r_min, max=r_max, nbins=N_bins)
    EDGE_r.append(star_prof['rbins'])
    EDGE_rho.append(DM_prof['density'] + gas_prof['density'] + star_prof['density'])

    hlr = pynbody.analysis.luminosity.half_light_r(s)

    with open('./files/CHIMERA_massive/rho_%s.txt' % h.timestep.extension, 'w') as f:
      np.savetxt(f, np.transpose([EDGE_r[-1], EDGE_rho[-1]]))
    with open('./files/CHIMERA_massive/hlr.txt', 'a') as f:
      f.write('%8.3f\n' % hlr)

  # Fit a profile to every single timestep.
  # - Fit a raw profile.
  # - Find the nearest gamma in multiples of 1/4.
  # - Refit with this gamma.
  # - Additionally, mass must be conserved. Keep the mass from the very first fit!
  #--------------------------------------------------------------------------
  fit_range = (EDGE_r[-1] > fit_min) & (EDGE_r[-1] <= fit_max)
  weights = np.sqrt(np.arange(len(EDGE_r[-1][fit_range])))

  # Perform the fit:
  # First pass to find best gamma:
  fit = Dehnen.fit(EDGE_rho[-1][fit_range], fit_params, r=EDGE_r[-1][fit_range], weights=weights)
  fixed_gamma = round(fit.best_values['gamma']*4)/4.
  fixed_fit_params.add('gamma', value=fixed_gamma, vary=False)
  # Second pass to find fit using best gamma:
  fit = Dehnen.fit(EDGE_rho[-1][fit_range], fixed_fit_params, r=EDGE_r[-1][fit_range], weights=weights)
  # Lock down Mg if this is the z=0 step:
  if h.timestep.extension==output:
    fit_params.add('log_Mg', value=fit.best_values['log_Mg'], vary=False)
    fixed_fit_params.add('log_Mg', value=fit.best_values['log_Mg'], vary=False)
  gammas.append(fit.best_values['gamma'])
  rs.append(10**fit.best_values['log_rs'])

  h = h.previous
  if h==None:
    break

gammas = np.array(gammas)
rs = np.array(rs)
Mg = 10**fit.best_values['log_Mg']

# Plot the evolution of the fit parameters:
fs = 14
fig, ax = plt.subplots(figsize=(6, 6))
#ax.plot(EDGE_t[:len(gammas)], gammas)
ax.plot(EDGE_t[:len(gammas)], rs)

# Find the most reprentative profile out of five time epochs:
# 1, 2, 4, 6, 13.8
EDGE_t = np.array(EDGE_t)
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
