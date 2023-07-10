import numpy as np

# Nbody6 path and simulation identifier:
#---------------------------------------------------------------------
path = '/user/HS301/m18366/EDGE_codes/GC_mock_obs/Nbody6_sims/'
sim_name = 'pilot'

# EDGE GC properties, units specified by astropy:
#---------------------------------------------------------------------
GC_pos = np.array([39.02408522, -205.59232791, -60.21901587]) # [pc]
GC_vel = np.array([-7.57310427, 1.39743804, -12.93909834]) # [km/s]
GC_mass = 7321.088 # [Msol]
GC_hlr = 3.533 # [pc]
GC_Z = -1.868 # [Logged]
GC_birthtime = 0.8535 # [Gyr]

# EDGE snapshot:
#---------------------------------------------------------------------
EDGE_path = '/vol/ph/astro_data/shared/morkney/EDGE/'
EDGE_sim_name = 'Halo1459_fiducial_hires'
EDGE_output = 'output_00020'
EDGE_halo = 1
