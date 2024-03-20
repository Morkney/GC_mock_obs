import numpy as np

# Select simulation suite:
#---------------------------------------------------------------------
# SUITES:
# __________________________________________________________________________________________________________________________
#| (0) DM         | (1) Full          | (2) Fantasy_cores   | (3) DM_compact | (4) Full_compact | (5) Fantasy_cores_compact |
#| Fit to DM only | Fit to all matter | Forced gamma=0 fits | Adjusted hmr   | Adjusted hmr     | Adjusted hmr              |
#| Incomplete     | Incomplete        | Incomplete          | Incomplete     | Incomplete       | Incomplete                |
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
suites = ['DM', 'Full', 'Fantasy_cores', 'DM_compact', 'Full_compact', 'Fantasy_cores_compact']
suite = suites[2]
print(suite)
path = '/vol/ph/astro_data/shared/morkney2/GC_mock_obs/'
#---------------------------------------------------------------------

# Simulation and tangos paths:
#---------------------------------------------------------------------
EDGE_path = {'EDGE':'/vol/ph/astro_data/shared/morkney/EDGE/', \
             'CHIMERA':'/vol/ph/astro_data/shared/etaylor/CHIMERA/'}
TANGOS_path = {'EDGE':'/vol/ph/astro_data/shared/morkney/EDGE/tangos/', \
               'CHIMERA':'/vol/ph/astro_data/shared/etaylor/CHIMERA/'}
#---------------------------------------------------------------------
