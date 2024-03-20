import numpy as np

import default_setup
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.ion()

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

# Make a normal looking plot:
#--------------------------------------------------------------
fs = 10
fig, ax = plt.subplots(figsize=(12, 4), ncols=3, gridspec_kw={'wspace':0.3})

masses = np.logspace(np.log10(2e3), np.log10(1e6), 100000)
r_half = young_SC_fit(masses)
r_half = add_scatter(r_half, sigma=0.735)

bins = [np.logspace(3, 6, 100), np.logspace(-1, 2, 100)]
ax[0].hist2d(masses, r_half, bins=bins, norm=LogNorm())
ax[0].loglog()

ax[0].set_xlabel(r'Initial mass [M$_{\odot}$]', fontsize=fs)
ax[0].set_ylabel(r'Initial half-mass radius [pc]', fontsize=fs)

ax[0].set_title('Fit to 1-10 Myr with scatter', fontsize=fs)
#--------------------------------------------------------------

# Compare with the EDGE SC sizes and masses:
#--------------------------------------------------------------
data1 = np.genfromtxt('./files/GC_property_table.txt', unpack=True, skip_header=2, dtype=None)
data2 = np.genfromtxt('./files/GC_property_table_CHIMERA.txt', unpack=True, skip_header=2, dtype=None)
data3 = np.genfromtxt('./files/GC_property_table_CHIMERA_massive.txt', unpack=True, skip_header=2, dtype=None)

data = np.array(list(data1) + list(data2) + list(data3))

GC_pos = np.array([[data[i][1], data[i][2], data[i][3]] for i in range(len(data))]) # kpc
GC_Rg = np.linalg.norm(GC_pos, axis=1) # kpc
GC_hlr = np.array([data[i][10] for i in range(len(data))]) # pc
GC_mass = 10**np.array([data[i][9] for i in range(len(data))]) # Msol
GC_Z = np.array([data[i][7] for i in range(len(data))]) # dec
EDGE_sim_name = np.array([data[i][11].decode("utf-8") for i in range(len(data))])
#--------------------------------------------------------------

# Plot the distributions:
#--------------------------------------------------------------
sims = [EDGE_sim_name[i] for i in sorted(np.unique(EDGE_sim_name, return_index=True)[1])]
colours = ['fuchsia', 'black', 'goldenrod', 'blueviolet', 'mediumseagreen', 'orangered', 'dodgerblue']
for i, (sim, colour) in enumerate(zip(sims, colours)):
  select = EDGE_sim_name == sim
  ax[1].scatter(GC_mass[select], GC_hlr[select], s=10, facecolor='None', edgecolor=colour, zorder=100-i, label=sim)
ax[1].loglog()

ax[1].set_xlabel(r'Initial mass [M$_{\odot}$]', fontsize=fs)
ax[1].set_ylabel(r'Initial half-mass radius [pc]', fontsize=fs)

ax[1].axhline(np.exp(np.mean(np.log(GC_hlr))))
ax[1].set_title('Raw EDGE data', fontsize=fs)

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

for i, (sim, colour) in enumerate(zip(sims, colours)):
  select = EDGE_sim_name == sim
  ax[2].scatter(GC_mass[select], new_GC_hlr[select], s=10, facecolor='None', edgecolor=colour, zorder=100-i, label=sim)
ax[2].loglog()

ax[2].set_xlabel(r'Initial mass [M$_{\odot}$]', fontsize=fs)
ax[2].set_ylabel(r'Initial half-mass radius [pc]', fontsize=fs)

ax[2].axhline(np.exp(np.mean(np.log(new_GC_hlr))))

ax[2].legend(loc='upper right', fontsize=fs-4)
ax[2].set_title('Adjusted EDGE data', fontsize=fs)
#--------------------------------------------------------------

for axes in fig.get_axes():
  axes.tick_params(which='both', axis='both', labelsize=fs-2)
  axes.set_xlim(*bins[0][[0,-1]])
  axes.set_ylim(*bins[1][[0,-1]])
  axes.set_aspect(np.diff(np.log10(axes.get_xlim())) / np.diff(np.log10(axes.get_ylim())))
