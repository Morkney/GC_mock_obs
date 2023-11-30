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

# The fit:
#--------------------------------------------------------------------------
# Standard 1002019651.1852752 1.4162238695676699 1.3435081502979234 1.0902598152276848 0.9046771807258164 0.9016080140166572 0.9065929534761556 0 0 0 0 0 0
# Massive 3942889189.1784205 4.947373260961169 4.351505567029788 1.9845962406496875 0.20000000000000004 0.20000000001313173 0.20000000000000004 0 0 0 0 0 0
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
r_min = 0.02
r_max = 3
fit_min = 0.05
fit_max = 2

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

priors['log_Mg']['guess'] = 9.5
priors['log_Mg']['min'] = 7.5
priors['log_Mg']['max'] = 11

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
hlr = np.loadtxt('./files/CHIMERA_massive/hlr.txt')
i = 0

while True:

  print('>    %s' % h.timestep.extension)
  if h.timestep.extension == 'output_00006': break

  data = np.loadtxt('./files/CHIMERA_massive/rho_%s.txt' % h.timestep.extension, unpack=True)
  EDGE_r.append(data[0])
  EDGE_rho.append(data[1])
  EDGE_t.append(h.calculate('t()'))

  hlr_ = min(hlr[i], 1.534)
  fit_min = max(hlr_, 0.03)
  fit_range = (EDGE_r[-1] > fit_min) & (EDGE_r[-1] <= fit_max)
  if np.sum(fit_range) < 1:
    rs.append(10**fit.best_values['log_rs'])
    h = h.previous
    i += 1
    continue

  weights = np.sqrt(np.arange(len(EDGE_r[-1][fit_range])))
  fit = Dehnen.fit(EDGE_rho[-1][fit_range], fit_params, r=EDGE_r[-1][fit_range])#, weights=weights)

  # Lock down Mg if this is the z=0 step:
  if h.timestep.extension==output:
    fit_params.add('log_Mg', value=fit.best_values['log_Mg'], vary=False)
  rs.append(10**fit.best_values['log_rs'])

  h = h.previous
  i += 1
  if h==None:
    break

rs = np.array(rs)
Mg = 10**fit.best_values['log_Mg']
EDGE_t = np.array(EDGE_t)

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
t_epochs = [(0.1, 0.9), (0.6, 1.4), (1.5, 2.5), (3, 5), (5, 7), (12, 14)]
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
