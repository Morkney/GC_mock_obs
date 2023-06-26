from config import *

import numpy as np

file = path + sim_name + '/YBC_%s.dat' % sim_name
data = np.loadtxt(file, skiprows=1, unpack=True)
format = '%.3f %.3f %.3f %.3e %.3f %.3f %.3e' + ' %.3f' * (np.shape(data)[0] - 7)
np.savetxt(file, data.T, fmt=format)
