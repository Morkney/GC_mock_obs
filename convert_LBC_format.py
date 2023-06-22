import numpy as np

file = './files/projected_GC_mock_data.dat'
data = np.loadtxt(file, skiprows=1, unpack=True)
format = '%.3f %.3f %.3f %.3e %.3f %.3f %.3e %.3f %.3f %.3f %.3f'
file = './files/COCOA_GC_mock_data.dat'
np.savetxt(file, data.T, header='x/pc, y/pc, z/pc, log(Lsol), Teff/K, m/Msol, Z', fmt=format)
