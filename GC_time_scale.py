import numpy as np
import sys

# Time scaling "T*":
T_scale = None
#T_scale = 2.127
if T_scale is None:
  T_scale = float(sys.argv[1])

# Maximum age is 13.8 Gyr:
age_max = 13.8 * 1e3 # [Myr]
#age_max = 1 * 1e3 # [Myr]

# Desired number of timesteps:
steps = int(250 * age_max / 1e3)

# output frequency which reproduces this:
output_f = int(np.floor(age_max / (T_scale * steps)))
max_output = int(age_max / T_scale)

print()
print('For %i outputs over %.2f Gyr:' % (steps, age_max/1e3))
print('output frequency = %i' % output_f)
print('Max output = %i' % max_output)
print()

