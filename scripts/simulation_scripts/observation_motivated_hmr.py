import numpy as np
import tangos

import default_setup
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import LogNorm
plt.ion()

from scipy.stats import norm, gaussian_kde

# Marta paper:
# https://arxiv.org/pdf/2306.17701.pdf
#--------------------------------------------------------------
def constant_density(mass):
  return 3 * (mass/1e4)**(0.3)

def young_SC_fit(mass):
  # Brown & Gnedin (2021)
  r_half = 2.365 * (mass/1e4)**(0.180)
  r_half[mass >= 1e5] = 2.365 * (1e5/1e4)**(0.180)
  return r_half

def add_scatter(r_half, sigma=0.3):
  return np.random.lognormal(mean=np.log(r_half), sigma=sigma)
#--------------------------------------------------------------

# Design figure:
#--------------------------------------------------------------
fs = 10
fig, ax = plt.subplots(figsize=(12, 4), ncols=3, gridspec_kw={'wspace':0.3})

# Binning:
N_bins = 100
mass_bins = np.logspace(np.log10(2e3), np.log10(4e5), N_bins)
hmr_bins = np.logspace(-1, 2, N_bins)
#--------------------------------------------------------------

# Compare with the EDGE SC sizes and masses:
#--------------------------------------------------------------
# Load Ethan's property dict:
from config import *
data = load_data()

#####
# Discard el fuckbois:
data['Halo383_fiducial_early'].pop(24)
data['Halo383_Massive'].pop(148)
#####

def get_dict(key):
  return np.concatenate([[data[i][j][key] for j in list(data[i].keys())] for i in list(data.keys())])

# Parse GC properties:
GC_pos = get_dict('Galacto-centred position')
GC_vel = get_dict('Galacto-centred velocity')
GC_hlr = get_dict('3D half-mass radius')
GC_mass = np.array([np.sum(i) for i in get_dict('Stellar Mass')])
GC_Z = get_dict('Median Fe/H')
GC_birthtime = get_dict('Median birthtime')
EDGE_sim_name = np.concatenate([[i]*len(data[i].keys()) for i in list(data.keys())])
EDGE_output = get_dict('Output Number')
EDGE_halo = get_dict('Tangos Halo ID')
GC_ID = np.concatenate([[j for j in list(data[i].keys())] for i in list(data.keys())])

# Parse particle properties:
GC_masses = get_dict('Stellar Mass')
GC_metals = get_dict('Fe/H Values')
GC_births = get_dict('Birth Times')

# Read the initial mass from the pre-created files:
for n, (i, j, k) in enumerate(zip(EDGE_sim_name, GC_ID, EDGE_output)):
  ID = data[i][j]['Internal ID']
  with open('../../Nbody6_sims/%s_files/%s_output_%05d_%s.txt' % (suite, i, k, ID)) as f:
    f.readline()
    GC_mass[n] = float(f.readline())
#--------------------------------------------------------------

# Plot the distributions:
#--------------------------------------------------------------
sims = [EDGE_sim_name[i] for i in sorted(np.unique(EDGE_sim_name, return_index=True)[1])]
sims = np.array(sims)[[1,3,4,5,2,0]]
colours = ['fuchsia', 'black', 'goldenrod', 'blueviolet', 'orangered', 'mediumseagreen']
colours = ['#F0E442', '#56B4E9', '#D55E00', '#0072B2', '#009E73', '#000000']
colours = ['#DDCC77', '#CC6677', '#882255', '#332288', '#117733', '#88CCEE']
for i, (sim, colour) in enumerate(zip(sims, colours)):
  select = EDGE_sim_name == sim
  ax[1].scatter(GC_mass[select], GC_hlr[select], s=10, facecolor='None', edgecolor=colour, zorder=100-i, label=sim.replace('_hires', '').replace('_fiducial', ''))
ax[1].loglog()

ax[1].set_xlabel(r'Initial mass [M$_{\odot}$]', fontsize=fs)
#ax[1].set_ylabel(r'Initial half-mass radius [pc]', fontsize=fs)

ax[1].set_title('Raw EDGE data', fontsize=fs, va='top', y=0.925)

ax[1].legend(loc='lower center', fontsize=fs-2, ncol=2, handletextpad=0.2, columnspacing=1.)

# Find distribution according to Marta's setup, but maintaining similar ordering:
# Repeat 10000 times to find the typical values:
repeats = 10000
new_GC_hlr = np.empty([repeats, len(GC_hlr)])
for i in range(repeats):
  new_hlr = add_scatter(young_SC_fit(GC_mass), sigma=0.735) / young_SC_fit(GC_mass)
  sorted_new_hlr = np.argsort(new_hlr)
  sorted_GC_hlr = np.argsort(GC_hlr)
  new_GC_hlr[i][sorted_GC_hlr] = new_hlr[sorted_new_hlr] * young_SC_fit(GC_mass[sorted_GC_hlr])
new_GC_hlr = np.median(new_GC_hlr, axis=0)

# Enforce maximum and minimum restrictions on the dataset:
new_GC_hlr[new_GC_hlr > 10.] = 10.
new_GC_hlr[new_GC_hlr < 0.5] = 0.5
#new_GC_hlr[np.where((EDGE_sim_name=='Halo624_fiducial_hires') & (GC_ID==3))] *= 1.5
new_GC_hlr[np.where((EDGE_sim_name=='Halo624_fiducial_hires') & (GC_ID==1))] *= 1.5

for i, (sim, colour) in enumerate(zip(sims, colours)):
  select = EDGE_sim_name == sim
  ax[2].scatter(GC_mass[select], new_GC_hlr[select], s=10, facecolor='None', edgecolor=colour, zorder=100-i, label=sim)
ax[2].loglog()

ax[2].set_xlabel(r'Initial mass [M$_{\odot}$]', fontsize=fs)
#ax[2].set_ylabel(r'Initial half-mass radius [pc]', fontsize=fs)

ax[2].set_title('Adjusted EDGE data', fontsize=fs, va='top', y=0.925)

ax[2].axhspan(hmr_bins[0], 0.5, facecolor='lightgrey', alpha=0.5, zorder=0)
ax[2].axhspan(10, hmr_bins[-1], facecolor='lightgrey', alpha=0.5, zorder=0)
ax[2].text(0.95, 0.5, 'Restricted sizes', va='top', ha='right', fontsize=fs-2, color='grey', transform=ax[2].get_yaxis_transform())
#--------------------------------------------------------------

# Mock GCs:
#--------------------------------------------------------------
#masses = np.logspace(*np.log10(mass_bins[[0,-1]]), 100000)
#r_half = young_SC_fit(masses)
#r_half = add_scatter(r_half, sigma=0.735)

factor = 3
weights = np.ones_like(GC_mass)
neff = np.sum(weights)**2 / np.sum(weights**2)
p = gaussian_kde(np.log10(GC_mass), weights=weights, bw_method=neff**(-1/(1+factor)))
kde = p((np.log10(mass_bins[1:]) + np.log10(mass_bins[:-1])) / 2.)
kde /= np.sum(kde)
pdf = np.cumsum(kde)
n_guesses = np.random.rand(1000000)
mass_points = np.sqrt(mass_bins[1:]*mass_bins[:-1])
masses = np.interp(n_guesses, pdf, mass_points)
r_half = young_SC_fit(masses)
r_half = add_scatter(r_half, sigma=0.735)

ax[0].hist2d(masses, r_half, bins=[mass_bins, hmr_bins], norm=LogNorm(), cmap=cm.Greys)
ax[0].loglog()

ax[0].set_xlabel(r'Initial mass [M$_{\odot}$]', fontsize=fs)
ax[0].set_ylabel(r'Initial half-mass radius [pc]', fontsize=fs)

ax[0].set_title('Fit to 1-10 Myr with scatter\n(Brown & Gnedin, 2021)', fontsize=fs, va='top', y=0.925)
#--------------------------------------------------------------

for axes in fig.get_axes():
  #axes.label_outer()
  axes.tick_params(which='both', axis='both', labelsize=fs-2)
  axes.set_xlim(*mass_bins[[0,-1]])
  axes.set_ylim(*hmr_bins[[0,-1]])
  axes.set_aspect(np.diff(np.log10(axes.get_xlim())) / np.diff(np.log10(axes.get_ylim())))

# Add histograms to the exterior axes:
#--------------------------------------------------------------
def get_KDE_1D(data, weights, bins, factor=3.):
  neff = np.sum(weights)**2 / np.sum(weights**2)
  p = gaussian_kde(data, weights=weights, bw_method=neff**(-1/(1+factor)))
  norm_to_data = len(data) * np.diff(bins) * np.mean(weights)
  kde = p((bins[1:] + bins[:-1]) / 2.) * norm_to_data
  return kde

# Append new axes:
for i, mass, hmr in zip(range(3), [masses, GC_mass, GC_mass], [r_half, GC_hlr, new_GC_hlr]):
  size = 0.225
  l, b, w, h = ax[i].get_position().bounds
  hax_x = fig.add_axes([l, b+h, w, h*size], zorder=-1)
  hax_y = fig.add_axes([l+w, b, w*size, h], zorder=-1)

  # Histogram:
  hist = np.histogram(mass, bins=mass_bins)
  hax_x.fill_between(x=mass_bins[1:], y1=0, y2=hist[0], color='silver', step='pre', lw=0, clip_on=False)
  hax_x.set_xlim(ax[i].get_xlim())
  hax_x.set_xscale('log')

  hist = np.histogram(hmr, bins=hmr_bins)
  hax_y.fill_betweenx(y=hmr_bins[1:], x1=0, x2=hist[0], color='silver', step='pre', lw=0, clip_on=False)
  hax_y.set_ylim(ax[i].get_ylim())
  hax_y.set_yscale('log')

  # Smoothed KDE:
  kde_x = get_KDE_1D(np.log10(mass), np.ones_like(mass), np.log10(mass_bins))
  mass_points = np.sqrt(mass_bins[1:]*mass_bins[:-1])
  hax_x.plot(mass_points, kde_x, 'k-', lw=1, clip_on=False)

  kde_y = get_KDE_1D(np.log10(hmr), np.ones_like(hmr), np.log10(hmr_bins))
  hmr_points = np.sqrt(hmr_bins[1:]*hmr_bins[:-1])
  hax_y.plot(kde_y, hmr_points, 'k-', lw=1, clip_on=False)
  if i==0:
    model_kde = kde_y
  else:
    hax_y.plot(model_kde/(model_kde.max()/kde_y.max()), hmr_points, 'r--', lw=1, clip_on=False)

  hax_x.set_ylim(ymin=0, ymax=kde_x.max()/(2./3.))
  hax_y.set_xlim(xmin=0, xmax=kde_y.max()/(2./3.))

  # Sigma lines:
  kwargs = {'lw':1, 'c':'k', 'solid_capstyle':'butt', 'clip_on':False}
  fractions = [50-34.1, 50, 50+34.1]
  sigmas = np.percentile(mass, fractions)
  d2a = (hax_x.transAxes + hax_x.transData.inverted()).inverted()
  ys = np.interp(sigmas, mass_points, kde_x)
  hax_x.axvline(sigmas[0], ymax=d2a.transform((0,ys[0]))[1], ls='--', **kwargs)
  hax_x.axvline(sigmas[1], ymax=d2a.transform((0,ys[1]))[1], ls='-', **kwargs)
  hax_x.axvline(sigmas[2], ymax=d2a.transform((0,ys[2]))[1], ls='--', **kwargs)

  sigmas = np.percentile(hmr, fractions)
  d2a = (hax_y.transAxes + hax_y.transData.inverted()).inverted()
  ys = np.interp(sigmas, hmr_points, kde_y)
  hax_y.axhline(sigmas[0], xmax=d2a.transform((ys[0],0))[0], ls='--', **kwargs)
  hax_y.axhline(sigmas[1], xmax=d2a.transform((ys[1],0))[0], ls='-', **kwargs)
  hax_y.axhline(sigmas[2], xmax=d2a.transform((ys[2],0))[0], ls='--', **kwargs)

  # Tick stylings:
  hax_x.set_xticks([])
  hax_x.set_yticks([])
  hax_y.set_xticks([])
  hax_y.set_yticks([])
  hax_x.set_axis_off()
  hax_y.set_axis_off()
#--------------------------------------------------------------

plt.savefig('../images/modified_hmr_plot.pdf', bbox_inches='tight')

# Write results to a dictionary file:
#--------------------------------------------------------------
'''
filename = '../files/adjusted_hmr.pk1'

import pickle
props = {}
for sim_name in np.unique(EDGE_sim_name):
  selection = EDGE_sim_name==sim_name
  props[sim_name] = {}
  for i, ID in enumerate(GC_ID[selection]):
    props[sim_name][ID] = new_GC_hlr[selection][i]

with open(filename, 'wb') as file:
  pickle.dump(props, file)
'''
#--------------------------------------------------------------
