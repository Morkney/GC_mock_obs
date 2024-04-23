from config import *

import numpy as np
import pynbody
pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
import tangos
import GC_functions as func
import sys
import os
import gc
import pickle

import matplotlib.pyplot as plt
plt.ion()

# Simulation choices:
#--------------------------------------------------------------------------
EDGE_sim_name = 'Halo383_Massive'
#--------------------------------------------------------------------------

# Load the simulation database:
#--------------------------------------------------------------------------
sim_type = 'CHIMERA' if '383' in EDGE_sim_name else 'EDGE'
EDGE_path = EDGE_path[sim_type]
TANGOS_path = TANGOS_path[sim_type]
tangos.core.init_db(TANGOS_path + EDGE_sim_name.split('_')[0] + '.db')
session = tangos.core.get_default_session()
#--------------------------------------------------------------------------

# Find the latest halo:
#--------------------------------------------------------------------------
output = tangos.get_simulation(EDGE_sim_name).timesteps[-1].extension
h = tangos.get_halo(EDGE_sim_name + '/' + output + '/' + 'halo_1')
halo = h.calculate('halo_number()') - 1
#--------------------------------------------------------------------------

# Find all haloes and outputs throughout time:
#--------------------------------------------------------------------------
#halos, paths = h.calculate_for_progenitors('halo_number()', 'path()')
#outputs = np.array([path.split('/')[1].split('/')[0] for path in paths])

halos = []
outputs = []
while h.previous != None:
  halos.append(h.calculate('halo_number()'))
  outputs.append(h.timestep.extension)
  h = h.previous
halos = np.array(halos)
outputs = np.array(outputs)

busted_outputs = ['output_00019', 'output_00029', 'output_00030', 'output_00031',
                  'output_00032', 'output_00035', 'output_00036', 'output_00037',
                  'output_00041', 'output_00044', 'output_00045']
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
i = 0
i = np.where(outputs == busted_outputs[i])[0][0]
h = tangos.get_halo(EDGE_sim_name + '/' + outputs[i] + '/' + 'halo_%i' % halos[i])
halo = h.calculate('halo_number()') - 1
print(f'>    {outputs[i]}')

s = pynbody.load(EDGE_path + EDGE_sim_name +'/'+ outputs[i], maxlevel=1)
s.physical_units()
s_h = s.halos()
s_h = s_h[halo]
s.g['pos']; s.d['pos']; s.s['pos']
s.g['mass']; s.d['mass']; s.s['mass']

# Centre:
cen = pynbody.analysis.halo.shrink_sphere_center(s_h.d, shrink_factor=0.95)
s['pos'] -= cen

plt.plot(s_h.d['x'], s_h.d['y'], 'k,', alpha=0.1)
plt.plot(0,0, 'rx')
plt.ylim(-3,3)
plt.xlim(-3,3)
plt.title(outputs[i])
plt.gca().set_aspect(1)
#--------------------------------------------------------------------------
