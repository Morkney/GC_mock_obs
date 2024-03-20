import numpy as np
import os, re, sys
import subprocess
from file_read_backwards import FileReadBackwards

# Retrieve the target directory:
directory = sys.argv[1] + '/out'

# Find the T* value:
with open(directory) as file:
  while True:
    line = file.readline()
    if 'T*' in line:
      break
T_star = float(line.split('T* =')[1].split('<M> =')[0])

# Find new timestepping:
age_max = 13.8 * 1e3
steps = int(250 * age_max / 1e3)
output_f = int(np.floor(2*age_max / (T_star * steps))) / 2
output_f = max(output_f, 0.5)
max_output = int(age_max / T_star)

# Return the timestepping parameters:
print(output_f, max_output)
