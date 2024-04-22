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

# Open dictionary:
#--------------------------------------------------------------------------
filename = path+'/scripts/files/CHIMERA_properties_dict.pk1'
if os.path.isfile(filename):
  with open(filename, 'rb') as file:
    props = pickle.load(file)
else:
  props = {}
#--------------------------------------------------------------------------

# Find the density profile evolution with time:
N_bins = 150 + 1
r_min = 0.02
r_max = 20
#--------------------------------------------------------------------------
i = 0
props[EDGE_sim_name] = {}
while True:

  halo = h.calculate('halo_number()') - 1
  output = h.timestep.extension
  props[EDGE_sim_name][output] = {}
  print(f'>    {output}')

  s = pynbody.load(EDGE_path + EDGE_sim_name +'/'+ output, maxlevel=20)
  s.physical_units()
  s_h = s.halos()
  s_h = s_h[halo]
  s.g['pos']; s.d['pos']; s.s['pos']
  s.g['mass']; s.d['mass']; s.s['mass']
  try:
    s.s['age'] = pynbody.analysis.ramses_util.get_tform(s)
  except:
    break

  # Centre:
  cen = pynbody.analysis.halo.shrink_sphere_center(s_h.d, shrink_factor=0.95)
  s['pos'] -= cen

  # Retrieve the profile:
  DM_prof = pynbody.analysis.profile.Profile(s.d, ndim=3, type='log', min=r_min, max=r_max, nbins=N_bins)
  gas_prof = pynbody.analysis.profile.Profile(s.g, ndim=3, type='log', min=r_min, max=r_max, nbins=N_bins)
  star_prof = pynbody.analysis.profile.Profile(s.s, ndim=3, type='log', min=r_min, max=r_max, nbins=N_bins)
  hlr = pynbody.analysis.luminosity.half_light_r(s)

  # Save to pickle dictionary:
  props[EDGE_sim_name][output]['t'] = pynbody.analysis.cosmology.age(s)
  props[EDGE_sim_name][output]['r'] = DM_prof['rbins']
  props[EDGE_sim_name][output]['DM_rho'] = DM_prof['density']
  props[EDGE_sim_name][output]['gas_rho'] = gas_prof['density']
  props[EDGE_sim_name][output]['star_rho'] = star_prof['density']
  props[EDGE_sim_name][output]['hlr'] = hlr

  # Add cumulative stellar histogram if this is the first step:
  if i==0:
    SFR_histogram = np.histogram(s.s['age'], range=[0, pynbody.analysis.cosmology.age(s)], bins=500)[0]
    props[EDGE_sim_name][output]['SFR_histogram'] = SFR_histogram

  del(s)
  del(s_h)
  gc.collect()

  h = h.previous
  i += 1

  if h is None:
    break
  with open(filename, 'wb') as file:
    pickle.dump(props, file)
#--------------------------------------------------------------------------
