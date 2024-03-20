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

DM_only = True

# The fits:
#--------------------------------------------------------------------------
# Fit with DM+gas+stars
# Halo1445 192474380.36046273 1.4884956245020209 0.9286634547921888 1.1030368592595123 0.8245889922520259 0.8458253028330703 0.8410221710629264 1.0 0.75 1.0 0.75 0.75 0.75
# Halo1459 187869975.68623495 0.8996810852229837 0.6213171830221742 0.6840549249545358 0.6997792085977168 0.6990678013797488 0.6957148245620229 1.0 0.75 0.75 0.75 0.75 0.75
# Halo600  714819341.9016784 3.1229024619035886 2.8584164179921077 2.4640192326470753 2.238263632239617 2.0150158546651804 2.04436606464254 1.0 1.0 1.0 1.0 1.0 1.0
# Halo605  1300407253.7867322 3.513737117188465 1.3368461549689457 2.370820995584988 2.3835600607216647 2.314759972099984 2.2847718128111487 1.0 0.5 1.0 1.0 1.0 1.0
# Halo624  691714733.1018294 2.187650824075687 1.8832524040766991 1.9618154180487481 1.331853101334287 1.3123061886661298 1.6899368375197799 1.0 1.0 1.0 0.75 0.75 1.0

# Fit with DM
# Halo1445 131048541.10433634 1.3221998702370144 0.7965407340518139 0.6851650054563079 0.6998775531039034 0.5634087335149933 0.5608759476737457 1.0 0.75 0.75 0.75 0.5 0.5
# Halo1459 160146057.9240216 0.8798510336678207 0.5855546185534086 0.519643464814186 0.5322363843672284 0.534742479364079 0.5336868355813636 1.0 0.75 0.5 0.5 0.5 0.5
# Halo600 818461420.5743201 3.637850227072299 3.3033833309353837 2.685867073445377 2.6158393479081594 1.6556971779491438 1.703841981461407 1.0 1.0 1.0 1.0 0.75 0.75
# Halo605 400236162.4303575 1.8923039701708846 0.8475159361994128 0.6485473405033116 0.6561629970273349 0.6444206685062946 0.6506258593428748 1.0 0.5 0.25 0.25 0.25 0.25
# Halo624 218690317.7846931 1.2534878513103824 0.658102560004776 0.5520876684195546 0.5311193338932869 0.5195402131841502 0.5019386357090192 1.0 0.5 0.25 0.25 0.25 0.25
#--------------------------------------------------------------------------

# Load the simulation database:
#--------------------------------------------------------------------------
EDGE_sim_name = 'Halo624_fiducial_hires'
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
if DM_only:
  EDGE_t, EDGE_rho, EDGE_r = h.calculate_for_progenitors('t()', 'dm_density_profile+gas_density_profile+star_density_profile', 'rbins_profile')
else:
  EDGE_t, EDGE_rho, EDGE_r = h.calculate_for_progenitors('t()', 'dm_density_profile', 'rbins_profile')

# Rebin the density:
N_bins = 100+1
r_range = (0.02, 3)
fit_range = (0.03, 2)
r = np.logspace(*np.log10(r_range), N_bins)
for i in range(len(EDGE_t)):
  EDGE_r[i], EDGE_rho[i] = func.rebin(EDGE_r[i], EDGE_rho[i], r)
fit_range = (EDGE_r[0] > fit_range[0]) & (EDGE_r[0] <= fit_range[1])
#--------------------------------------------------------------------------

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
#--------------------------------------------------------------------------

# Add these constraints to the Dehnen model:
#--------------------------------------------------------------------------
Dehnen = Model(func.Dehnen_profile)
fit_params = Parameters()
for param in params:
  fit_params.add(param, value=priors[param]['guess'], min=priors[param]['min'], max=priors[param]['max'])
fixed_fit_params = fit_params.copy()
weights = np.sqrt(np.arange(len(EDGE_r[0][fit_range])))
#--------------------------------------------------------------------------

# Loop over each step and perform the fit:
#--------------------------------------------------------------------------
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
#--------------------------------------------------------------------------

# Check a fit:
#--------------------------------------------------------------------------
i = 0
fs = 14
fig, ax = plt.subplots(figsize=(6, 6))
ax.loglog(EDGE_r[i], EDGE_rho[i], 'k', lw=2)
ax.loglog(EDGE_r[i], func.Dehnen_profile(EDGE_r[i], np.log10(rs[i]), np.log10(Mg), gammas[i]), ls='--', lw=1)
#--------------------------------------------------------------------------

# Plot the evolution of the fit parameters:
#--------------------------------------------------------------------------
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
#--------------------------------------------------------------------------

#### The below section is brainstorming on a smoother transition. ####
# I can use a spline! (?)
#--------------------------------------------------------------------------
'''
def delta_x_t(t, x_i, v_i, x_f, v_f, t_i, t_f):
  # https://arxiv.org/pdf/2402.17889.pdf
  A_x = x_i
  B_x = v_i
  C_x = 3*(x_f-x_i)*(t_f-t_i)**(-2) - (2*v_i+v_f)*(t_f-t_i)**(-1)
  D_x = 2*(x_i-x_f)*(t_f-t_i)**(-3) + (v_i+v_f)*(t_f-t_i)**(-2)
  return A_x + B_x*(t-t_i) + C_x*(t-t_i)**2 + D_x*(t-t_i)**3

# Create a cubic spline to smooth the parameter evolution:
t_epoch = np.array([np.mean(t_epoch) for t_epoch in t_epochs])
rs_vel = np.gradient(rs_epoch) / np.gradient(t_epoch)
gamma_vel = np.gradient(gamma_epoch) / np.gradient(t_epoch)

for i in range(len(t_epoch)-1):
  t_arr = np.linspace(t_epoch[i], t_epoch[i+1], 10)
  rs_arr = delta_x_t(t_arr, rs_epoch[i], rs_vel[i], rs_epoch[i+1], rs_vel[i+1], t_epoch[i], t_epoch[i+1])
  ax.plot(t_arr, rs_arr, 'k-')

t_epoch = [0, 1, 2, 4, 6, 13.8]
a = np.arange(len(t_epoch))
av = np.gradient(alphas) / np.gradient(t_epoch)

i = 0
t_spline = np.linspace(t_epoch[i], t_epoch[i+1], 100)
a_spline = delta_x_t(t_spline, a[i], av[i], a[i+1], av[i+1], t_epoch[i], t_epoch[i+1])

def sigmoid(x, beta=3.):
  return 1 / (1 + (x / (1-x))**(-beta))

def super_interp(params, alpha):
  alpha = sigmoid(alpha)
  return alpha*param[0] + (1-alpha)*param[1]

# Need a good way to visualise the impact of the transition!

# Need more thought on this subject...
'''
#--------------------------------------------------------------------------

# Plot goodness of fit:
#--------------------------------------------------------------------------
def sigmoid(x, beta=3.):
  return 1 / (1 + (x / (1-x))**(-beta))

def interp(param, alpha):
  return alpha*param[0] + (1-alpha)*param[1]

def interp_profile(r, t, rs_epoch, gamma_epoch):
  if (t < 1):
    profile0 = func.Dehnen_profile(r, np.log10(rs_epoch[0]), np.log10(Mg), gamma_epoch[0])
    profile1 = func.Dehnen_profile(r, np.log10(rs_epoch[1]), np.log10(Mg), gamma_epoch[1])
    alpha = (t) / (1.-0.)
    profile = interp([profile1, profile0], sigmoid(alpha))
  elif (t < 2) & (t >= 1):
    profile0 = func.Dehnen_profile(r, np.log10(rs_epoch[1]), np.log10(Mg), gamma_epoch[1])
    profile1 = func.Dehnen_profile(r, np.log10(rs_epoch[2]), np.log10(Mg), gamma_epoch[2])
    alpha = (t-1.) / (2.-1.)
    profile = interp([profile1, profile0], sigmoid(alpha))
  elif (t < 4) & (t >= 2):
    profile0 = func.Dehnen_profile(r, np.log10(rs_epoch[2]), np.log10(Mg), gamma_epoch[2])
    profile1 = func.Dehnen_profile(r, np.log10(rs_epoch[3]), np.log10(Mg), gamma_epoch[3])
    alpha = (t-2.) / (4.-2.)
    profile = interp([profile1, profile0], sigmoid(alpha))
  elif (t < 6) & (t >= 4):
    profile0 = func.Dehnen_profile(r, np.log10(rs_epoch[3]), np.log10(Mg), gamma_epoch[3])
    profile1 = func.Dehnen_profile(r, np.log10(rs_epoch[4]), np.log10(Mg), gamma_epoch[4])
    alpha = (t-4.) / (6.-4.)
    profile = interp([profile1, profile0], sigmoid(alpha))
  elif (t < 14):
    profile0 = func.Dehnen_profile(r, np.log10(rs_epoch[4]), np.log10(Mg), gamma_epoch[4])
    profile1 = func.Dehnen_profile(r, np.log10(rs_epoch[5]), np.log10(Mg), gamma_epoch[5])
    alpha = (t-6.) / (13.8-6.)
    profile = interp([profile1, profile0], sigmoid(alpha))

  return profile

fs = 12
fig, ax = plt.subplots(figsize=(8,4), ncols=2)

# Get a range of colours that correspond to the edge times:
from scipy.ndimage import median_filter

def NormaliseData(data):
  return (data - np.min(data)) / (np.max(data) - np.min(data))
colors = cm.coolwarm(NormaliseData(EDGE_t))

'''
rho_with_t = np.zeros_like(EDGE_t)
for i, (r, t, color) in enumerate(zip(EDGE_r, EDGE_t, colors)):
  r = np.logspace(np.log10(0.02), np.log10(20), 100)
  profile = interp_profile(r, t, rs_epoch, gamma_epoch)
  ax[0].loglog(r, profile, lw=1, color=color)
  rho_with_t[i] = profile[32]
'''

rho_with_t = np.zeros_like(EDGE_t)
for i, (r, t, color) in enumerate(zip(EDGE_r, EDGE_t, colors)):
  #r = np.logspace(np.log10(0.02), np.log10(20), 100)
  profile = func.Dehnen_profile(r, np.log10(rs[i]), np.log10(Mg), gammas[i])
  ax[0].loglog(r, profile, lw=1, color=color, zorder=1000-i)
  rho_with_t[i] = profile[32]

ax[0].set_xlim(*r_range)

ax[1].semilogy(EDGE_t, rho_with_t)

print(Mg, ' '.join(['%s' % i for i in rs_epoch]), ' '.join(['%s' % i for i in gamma_epoch]))
#--------------------------------------------------------------------------
