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
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt, matplotlib.patches as patches
plt.ion()

from scipy.ndimage import median_filter
from scipy.ndimage import percentile_filter
from scipy.ndimage import gaussian_filter

import pickle

# Simulation choices:
#--------------------------------------------------------------------------
profile_types = ('DM', 'Full', 'fantasy_core')
profile_type = profile_types[1]
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
if profile_type == 'DM':
  EDGE_t, EDGE_rho, EDGE_r = h.calculate_for_progenitors('t()', 'dm_density_profile', 'rbins_profile')
else:
  EDGE_t, EDGE_rho, EDGE_r = h.calculate_for_progenitors('t()', 'dm_density_profile+gas_density_profile+star_density_profile', 'rbins_profile')

# Rebin the density:
N_bins = 150+1
r_range = (0.02, 20)
fit_range = [0.035, 3]
r = np.logspace(*np.log10(r_range), N_bins)
for i in range(len(EDGE_t)):
  EDGE_r[i], EDGE_rho[i] = func.rebin(EDGE_r[i], EDGE_rho[i], r)
fit_range_arr = (EDGE_r[0] > fit_range[0]) & (EDGE_r[0] <= fit_range[1])
#--------------------------------------------------------------------------

# Initial fit parameters:
#--------------------------------------------------------------------------
params = ['log_rs', 'log_Mg', 'gamma']
priors = {}
for param in params:
  priors[param] = {}

priors['log_rs']['guess'] = np.log10(0.1)
priors['log_rs']['min'] = -3
priors['log_rs']['max'] = 2
priors['log_rs']['vary'] = True
#priors['log_rs']['guess'] = np.log10(0.4)
#priors['log_rs']['min'] = np.log10(0.2)
#priors['log_rs']['max'] = 2

priors['log_Mg']['guess'] = np.log10(h['M200c'])
priors['log_Mg']['min'] = 7.5
priors['log_Mg']['max'] = 10
priors['log_Mg']['vary'] = True

if profile_type == 'fantasy_core':
  priors['gamma']['guess'] = 0.0
  priors['gamma']['min'] = None
  priors['gamma']['max'] = None
  priors['gamma']['vary'] = False
else:
  priors['gamma']['guess'] = 0.5
  priors['gamma']['min'] = 0.0
  priors['gamma']['max'] = 1.0
  priors['gamma']['vary'] = True
#--------------------------------------------------------------------------

# Add these constraints to the Dehnen model:
#--------------------------------------------------------------------------
Dehnen = Model(func.Dehnen_profile)
fit_params = Parameters()
for param in params:
  fit_params.add(param, value=priors[param]['guess'], min=priors[param]['min'], max=priors[param]['max'], vary=priors[param]['vary'])
fixed_fit_params = fit_params.copy()
weights = np.sqrt(np.arange(len(EDGE_r[0][fit_range_arr])))
#--------------------------------------------------------------------------

# Loop over each step and perform the fit:
#--------------------------------------------------------------------------
gammas = []
rs = []
for j in range(2):
  for i in range(len(EDGE_t)):
    print('>    %i' % i)

    # For fantasy cores, base fit on the half light radius:
    #--------------------------------------------------------------------------
    if profile_type == 'fantasy_core':
      # Crop the fit range to the stellar half light:
      # I need a much better and more consistent method here! The half radius can vary stochastically and by a lot...
      if i == 0:
        new_fit_min = np.min(h.calculate_for_progenitors('stellar_3D_halflight', nmax=3))
        fit_range[0] = new_fit_min
      fit_range_arr = (EDGE_r[0] > fit_range[0]) & (EDGE_r[0] <= fit_range[1])
      # Use a prior fit if the stellar half light excludes the whole radial range:
      if np.sum(fit_range_arr) < 1:
        rs.append(10**fit.best_values['log_rs'])
        gammas.append(fit.best_values['gamma'])
        continue
      # Perform fit:
      fit = Dehnen.fit(EDGE_rho[i][fit_range_arr], fit_params, r=EDGE_r[i][fit_range_arr])
    #--------------------------------------------------------------------------

    # Otherwise, base fit on quarter-integer of gamma:
    #--------------------------------------------------------------------------
    else:
      # First pass to find best gamma:
      if j==0:
        fit = Dehnen.fit(EDGE_rho[i][fit_range_arr], fit_params, r=EDGE_r[i][fit_range_arr], weights=weights)
        fixed_gamma = round(fit.best_values['gamma']*4)/4.
      elif j==1:
        fixed_gamma = gammas[i]
      fixed_fit_params.add('gamma', value=fixed_gamma, vary=False)

      # Second pass to find fit using best gamma:
      fit = Dehnen.fit(EDGE_rho[i][fit_range_arr], fixed_fit_params, r=EDGE_r[i][fit_range_arr], weights=weights)
    #--------------------------------------------------------------------------

    # Lock down Mg if this is the z=0 step:
    #--------------------------------------------------------------------------
    if i==0:
      fit_params.add('log_Mg', value=fit.best_values['log_Mg'], vary=False)
      fixed_fit_params.add('log_Mg', value=fit.best_values['log_Mg'], vary=False)
    #--------------------------------------------------------------------------

    # Update parameter arrays:
    #--------------------------------------------------------------------------
    rs.append(10**fit.best_values['log_rs'])
    if j==0:
      gammas.append(fit.best_values['gamma'])
    #--------------------------------------------------------------------------

  # Smooth the gammas and refit:
  #--------------------------------------------------------------------------
  if (j==0) & (profile_type!='fantasy_core'):
    gammas_old = np.array(gammas.copy())
    gammas = np.array(gammas)
    if '6' in EDGE_sim_name:
      gammas = percentile_filter(gammas, 60, 12) # 10
      gammas = median_filter(gammas, 5) # 5
    else:
      gammas = percentile_filter(gammas, 60, 10)
      gammas = median_filter(gammas, 5)
    rs = []
  else:
    gammas_old = np.array(gammas.copy())
    break
  #--------------------------------------------------------------------------

gammas = np.array(gammas)
rs = np.array(rs)
Mg = 10**fit.best_values['log_Mg']
#--------------------------------------------------------------------------

# Remove sudden spikes and smooth the density via the scale radius parameter:
#--------------------------------------------------------------------------
smoothed_rs = median_filter(rs, 5)
rs2 = rs.copy()
spikes = ~np.isclose(smoothed_rs, rs, rtol=0.1) # Increase rtol to include more spikes.
rs2[spikes] = smoothed_rs[spikes]

# For each distinct gamma sequence, smooth the sequence in linear time:
gamma_steps = gammas[np.append([0], np.where(np.diff(gammas))[0]+1)]
for gamma in gamma_steps:
  if np.sum(gammas==gamma) <= 3:
    continue
  t_temp = np.linspace(*EDGE_t[gammas==gamma][[-1,0]], int(50*np.diff(EDGE_t[gammas==gamma][[-1,0]])))
  rs_temp = np.interp(t_temp, np.flip(EDGE_t[gammas==gamma]), np.flip(rs2[gammas==gamma]))
  rs_temp = gaussian_filter(rs_temp, mode='nearest', sigma=5)
  rs2[gammas==gamma] = np.flip(np.interp(np.flip(EDGE_t[gammas==gamma]), t_temp, rs_temp))

# Second pass:
smoothed_rs = median_filter(rs2, 5)
spikes = ~np.isclose(smoothed_rs, rs2, rtol=0.1) # Increase rtol to include more spikes.
rs2[spikes] = smoothed_rs[spikes]
#--------------------------------------------------------------------------

# Plot the evolution of the fit parameters:
#--------------------------------------------------------------------------
# Get a range of colours that correspond to the edge times:
def NormaliseData(data):
  return (data - np.min(data)) / (np.max(data) - np.min(data))

# Compare the raw and smoothed data:
fs = 10
fig, ax = plt.subplots(figsize=(8, 8), ncols=2, nrows=2, gridspec_kw={'wspace':0.3, 'hspace':0.2})

colors = cm.coolwarm(NormaliseData(EDGE_t**(1/4.)))
rho_at_r = 0.05 # [kpc]
rho_with_t_fit = np.zeros_like(EDGE_t)
rho_with_t_smoothed = np.zeros_like(EDGE_t)
rho_with_t_raw = np.zeros_like(EDGE_t)
r = np.logspace(np.log10(0.02), np.log10(20), 200)
for i, (t, color) in enumerate(zip(EDGE_t, colors)):
  rho_with_t_raw[i] = np.interp(rho_at_r, EDGE_r[i], EDGE_rho[i])
  profile = func.Dehnen_profile(r, np.log10(rs[i]), np.log10(Mg), gammas[i])
  rho_with_t_fit[i] = np.interp(rho_at_r, r, profile)
  profile = func.Dehnen_profile(r, np.log10(rs2[i]), np.log10(Mg), gammas[i])
  rho_with_t_smoothed[i] = np.interp(rho_at_r, r, profile)
  ax[0,0].loglog(r, profile, lw=1, color=color, zorder=1000-i)

ax[1,0].plot(EDGE_t[:len(gammas_old)], gammas_old, 'k')
ax[1,0].plot(EDGE_t[:len(gammas)], gammas, 'r')
ax[1,1].plot(EDGE_t[:len(rs)], rs, 'k')
ax[1,1].plot(EDGE_t[:len(rs2)], rs2, 'r')

ax[1,0].set_yticks([0, 0.25, 0.5, 0.75, 1])
ax[1,1].set_yscale('log')
ax[1,0].set_xscale('log')
ax[1,1].set_xscale('log')

ax[1,0].set_xlabel('Time [Gyr]', fontsize=fs)
ax[1,1].set_xlabel('Time [Gyr]', fontsize=fs)
ax[1,0].set_ylabel('Gamma from fit', fontsize=fs)
ax[1,1].set_ylabel('Scale radius from fit [kpc]', fontsize=fs)

# Grey-out the regions that are not being fit to:
ax[0,0].set_xlim(*r[[0,-1]])
ax[0,1].set_ylim(np.nanpercentile(rho_with_t_smoothed, [1,99])*np.array([0.6,1.5]))
ax[0,0].axvspan(0, fit_range[0], facecolor='whitesmoke', zorder=0)
ax[0,0].axvspan(fit_range[1], 100, facecolor='whitesmoke', zorder=0)
ax[0,0].axvline(rho_at_r, c='silver', ls='--', lw=1)

ax[0,1].semilogy(EDGE_t, rho_with_t_raw, 'grey', label='EDGE')
ax[0,1].semilogy(EDGE_t, rho_with_t_fit, 'k', label='Raw fit')
ax[0,1].semilogy(EDGE_t, rho_with_t_smoothed, 'r', label='Smooth fit')
ax[0,1].legend(fontsize=fs-4)

ax[0,1].set_xscale('log')

ax[0,0].set_ylabel(r'Density [M$_{\odot}\,{\rm kpc}^{-3}$]', fontsize=fs)
ax[0,0].set_xlabel('Radius [kpc]', fontsize=fs)
ax[0,1].set_ylabel(r'Density at $R_{\rm G}=%.2f\,$kpc [M$_{\odot}\,{\rm kpc}^{-3}$]' % rho_at_r, fontsize=fs)
ax[0,1].set_xlabel('Time [Gyr]', fontsize=fs)

for axes in np.ravel(fig.get_axes()):
  axes.tick_params(axis='both', which='both', labelsize=fs-2)

ax[0,0].plot(EDGE_r[0], EDGE_rho[0], 'k', zorder=1000, lw=1, ls='--', label=r'EDGE ($z=0$)')
#--------------------------------------------------------------------------

# Load the GC ICs and represent them:
#--------------------------------------------------------------------------
data = np.genfromtxt('./files/GC_property_table.txt', unpack=True, skip_header=2, dtype=None)

EDGE_sim_names = np.array([data[i][11].decode("utf-8") for i in range(len(data))])
this_sim = EDGE_sim_names == EDGE_sim_name

GC_pos = np.array([[data[i][1], data[i][2], data[i][3]] for i in range(len(data))])[this_sim] # kpc
GC_Rg = np.linalg.norm(GC_pos, axis=1)
GC_vel = np.array([[data[i][4], data[i][5], data[i][6]] for i in range(len(data))])[this_sim] # km s^-1
GC_birthtime = np.array([data[i][8] for i in range(len(data))])[this_sim] / 1e3 # Gyr

# Rug plots:
lim = ax[0,0].get_ylim()[0]
ax[0,0].plot(GC_Rg, [0.075]*len(GC_Rg), '|', color='k', label='GC ICs', transform=ax[0,0].get_xaxis_transform())
lim = ax[0,1].get_ylim()[0]
ax[0,1].plot(GC_birthtime, [0.075]*len(GC_birthtime), '|', color='k', transform=ax[0,1].get_xaxis_transform())

ax[0,0].legend(loc='upper right', fontsize=fs-4)
#--------------------------------------------------------------------------

# Save plot:
#--------------------------------------------------------------------------
fig.suptitle(EDGE_sim_name, fontsize=fs, y=0.925)
plt.savefig('./images/%s_evolution_%s.pdf' % (EDGE_sim_name, profile_type), bbox_inches='tight')
#--------------------------------------------------------------------------

# Make a new dictionary if one doesn't already exist, save the fits:
#--------------------------------------------------------------------------
filename = './files/host_profiles_dict.pk1'
if os.path.isfile(filename):
  with open(filename, 'rb') as file:
    props = pickle.load(file)
else:
  props = {}

EDGE_sim_name += '_' + profile_type
props[EDGE_sim_name] = {}
props[EDGE_sim_name]['Mg'] = Mg
props[EDGE_sim_name]['time'] = np.flip(np.concatenate([[100], EDGE_t, [0]]))
props[EDGE_sim_name]['rs'] = np.flip(np.concatenate([[rs2[0]], rs2, [rs2[-1]]]))
props[EDGE_sim_name]['gamma'] = np.flip(np.concatenate([[gammas[0]], gammas, [gammas[-1]]]))

# Save to dictionary:
with open(filename, 'wb') as file:
  pickle.dump(props, file)

# Also save to a txt file:
data = np.transpose([props[EDGE_sim_name]['time'], props[EDGE_sim_name]['rs'], props[EDGE_sim_name]['gamma']])
np.savetxt('./files/%s.txt' % EDGE_sim_name, data)
#--------------------------------------------------------------------------
