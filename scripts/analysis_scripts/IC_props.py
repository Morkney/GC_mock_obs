import numpy as np
import GC_functions as func
import sys
import os

import default_setup
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

# Load the GC ICs:
#--------------------------------------------------------------------------
data1 = np.genfromtxt(path + 'files/GC_property_table.txt', unpack=True, skip_header=2, dtype=None)
data2 = np.genfromtxt(path + 'files/GC_property_table_CHIMERA.txt', unpack=True, skip_header=2, dtype=None)
data3 = np.genfromtxt(path + 'files/GC_property_table_CHIMERA_massive.txt', unpack=True, skip_header=2, dtype=None)

data = np.array(list(data1) + list(data2) + list(data3))

GC_pos = np.array([[data[i][1], data[i][2], data[i][3]] for i in range(len(data))]) # kpc
GC_Rg = np.linalg.norm(GC_pos, axis=1) # kpc
GC_hlr = np.array([data[i][10] for i in range(len(data))]) # pc
GC_mass = 10**np.array([data[i][9] for i in range(len(data))]) # Msol
GC_Z = np.array([data[i][7] for i in range(len(data))]) # dec
EDGE_sim_name = np.array([data[i][11].decode("utf-8") for i in range(len(data))])
#--------------------------------------------------------------------------

# Plot result:
#--------------------------------------------------------------------------
fs = 12
fig, ax = plt.subplots(figsize=(6,6))

sims = [EDGE_sim_name[i] for i in sorted(np.unique(EDGE_sim_name, return_index=True)[1])]
colours = ['fuchsia', 'black', 'goldenrod', 'blueviolet', 'mediumseagreen', 'orangered', 'dodgerblue']
for i, (sim, colour) in enumerate(zip(sims, colours)):
  select = EDGE_sim_name == sim
  ax.scatter(GC_hlr[select], GC_mass[select]/2., s=10, facecolor='None', edgecolor=colour, zorder=100-i, label=sim)
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend(loc='upper right', fontsize=fs-4)

ax.tick_params(which='both', axis='both', labelsize=fs-2)
ax.set_xlabel(r'Half-light radius [pc]', fontsize=fs)
ax.set_ylabel(r'Half mass [M$_{\odot}$]', fontsize=fs)

ax.axvline(3, c='k', ls='--', lw=1)
ax.text(3*1.1, 0.95, 'EDGE resolution limit', fontsize=fs-4, rotation=90, va='top', ha='left', transform=ax.get_xaxis_transform())

# Add target regions:
ax.text(np.sqrt(0.8*5), 1.01, 'Krumholz 2019\nChoksi 2021', fontsize=fs-4, color='blue', va='bottom', ha='center', transform=ax.get_xaxis_transform())
ax.axvspan(xmin=0.8, xmax=5, facecolor='b', alpha=0.15)
ax.text(np.sqrt(0.1*0.3), 1.01, 'Leigh 2021', fontsize=fs-4, color='red', va='bottom', ha='center', transform=ax.get_xaxis_transform())
ax.axvspan(xmin=0.1, xmax=0.3, facecolor='r', alpha=0.15)

ylim = ax.get_ylim()
xlim = ax.get_xlim()
hmr_arr = np.logspace(*np.log10(xlim), 10)
'''
GC_rho = [1e4, 1e6]
half_mass = np.vstack(GC_rho) * (4/3.) * np.pi * hmr_arr**3
ax.fill_between(x=hmr_arr, y1=half_mass[0], y2=half_mass[1], facecolor='r', alpha=0.2)
'''

def text_angle(ax, x, y):
  return np.rad2deg(np.arctan2((y[-1]-y[0]) * (np.diff(np.log10(ax.get_xlim()))/np.diff(np.log10(ax.get_ylim())))[0], x[-1]-x[0]))

# Add lines of constant density:
ax.set_ylim(ylim)
ax.set_xlim(xlim)
GC_rho = 10.**np.arange(-10,10)
half_masses = np.vstack(GC_rho) * (4/3.) * np.pi * hmr_arr**3
for (rho, half_mass) in zip(GC_rho, half_masses):
  ax.plot(hmr_arr, half_mass, color='silver', ls='--', lw=1, zorder=0)
  yloc = 0.6
  xloc = 10**np.interp(np.log10(ylim[0])+(np.diff(np.log10(ylim)))*yloc, np.log10(half_mass), np.log10(hmr_arr)) * 0.9
  if (xloc > hmr_arr[0]*1.5) & (xloc < hmr_arr[-1]*0.75):
    ax.text(xloc, yloc, r'$\log\,\rho[{\rm M}_{\odot}\,{\rm pc}^{-3}]=%i$' % np.log10(rho), fontsize=fs-4, color='silver', zorder=0, \
            rotation=text_angle(ax, np.log10(hmr_arr), np.log10(half_mass)), va='center', ha='center', transform=ax.get_xaxis_transform())

#--------------------------------------------------------------------------
