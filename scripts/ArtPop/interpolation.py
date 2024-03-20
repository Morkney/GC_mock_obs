import numpy as np

from config import *
import GC_functions as func

import agama
from scipy.interpolate import make_interp_spline, CubicHermiteSpline
agama.setUnits(mass=1, length=1e-3, velocity=1)
timeunit = 0.97779222

def sigmoid(x, beta=3.):
  return 1 / (1 + (x / (1-x))**(-beta))

def spline_interpolation(s, s_interp):
  # Store the snapshot data in arrays:
  arrays = {}
  fields = ['mass', 'rdens', 'rg', 'hlr', 'shrink_centre', 'kstara', 'nbound', 'pos', 'vel']
  for field in fields:
    arrays[field] = np.array([i[field] for i in s])

  # Define a new resolution:
  orig_res = np.linspace(s[0]['age'], s[-1]['age'], len(s))
  spline_res = np.linspace(s_interp[0]['age'], s_interp[-1]['age'], len(s_interp))

  # Interpolate all necessary arrays to new resolution:
  for field in fields[:-4]:
    arrays[field] = make_interp_spline(orig_res, arrays[field], k=3)(spline_res)
  arrays['pos'] = CubicHermiteSpline(orig_res, arrays['pos'], arrays['vel']*timeunit)(spline_res)
  indices = np.floor(np.interp(spline_res, orig_res, np.arange(len(s)))).astype('int')
  arrays['kstara'] = arrays['kstara'][indices]
  arrays['nbound'] = arrays['nbound'][indices]

  # Replace orbit integration of unbound stars with csplines:
  for i in range(len(s_interp)):
    for field in fields[:-2]:
      s_interp[i][field] = arrays[field][i]
    #central_stars = ~s_interp[i]['nbound'] * (np.linalg.norm(s_interp[i]['pos'], axis=1) <= 1)
    s_interp[i]['pos'][~s_interp[i]['nbound']] = arrays['pos'][i][~s_interp[i]['nbound']]

  return s_interp

def orbit_interpolation(s, interpolated_steps, total_time):
  s_interp = []
  A = A = sigmoid(np.linspace(0, 1, interpolated_steps))
  for i in range(len(s)):

    # Make a spherically symmetric profile fit:
    rmax = s[i]['r'][s[i]['nbound']].max()
    data_snap = (np.concatenate([s[i]['pos'], s[i]['vel']], axis=1), s[i]['mass'])
    potential = agama.Potential(type='Multipole', particles=data_snap, gridSizeR=100, rmin=1e-3, rmax=rmax, symmetry='Spherical')

    # Integrate the orbits over the backward timestep and interpolate:
    if i > 0:
      orbits = agama.orbit(ic=data_snap[0], potential=potential, time=-total_time, trajsize=interpolated_steps)
      for j in range(1, interpolated_steps-1):
        pos_vel = np.vstack([k[1][j] for k in orbits])
        s_interp[-j]['pos'] = (1-A[j])*pos_vel[:,[0,1,2]] + A[j]*s_interp[-j]['pos']
        s_interp[-j]['vel'] = (1-A[j])*pos_vel[:,[3,4,5]] + A[j]*s_interp[-j]['vel']

    # Calculate orbits over the forward timestep:
    if i==len(s) - 1:
      snapshot = {}
      snapshot['age'] = s[i]['age']
      snapshot['pos'] = s[i]['pos']
      snapshot['vel'] = s[i]['vel']
      snapshot['shrink_centre'] = s[i]['shrink_centre']
      s_interp.append(snapshot)
    else:
      orbits = agama.orbit(ic=data_snap[0], potential=potential, time=total_time, trajsize=interpolated_steps)
      for j in range(interpolated_steps-1):
        snapshot = {}
        snapshot['age'] = s[i]['age'] + orbits[0][0][j]
        pos_vel = np.vstack([k[1][j] for k in orbits])
        snapshot['pos'] = pos_vel[:,[0,1,2]]
        snapshot['vel'] = pos_vel[:,[3,4,5]]
        snapshot['shrink_centre'] = s[i]['shrink_centre']
        s_interp.append(snapshot)
  return s_interp

def combined_interpolation(s, t_min, t_max, delta_t, stitch=True):

  # Take a time-slice of the simulation data:
  times = np.array([i['age'] for i in s])
  time_slice = (times >= t_min) & (times <= t_max)
  s_slice = [s[i].copy() for i in np.where(time_slice)[0]]

  # Define the new set of timesteps using delta_t:
  cadence_multiplier = round((s_slice[1]['age'] - s_slice[0]['age']) / delta_t)
  interpolated_steps = cadence_multiplier + 1
  total_time = s_slice[1]['age'] - s_slice[0]['age']

  # Upscale with orbit integration:
  print('Orbit interpolation...')
  s_slice = orbit_interpolation(s_slice, interpolated_steps, total_time)
  print('Spline interpolation...')
  s_slice = spline_interpolation(s, s_slice)

  # Replace section of 's' with 's_slice':
  if stitch:
    s = np.concatenate([s[:np.where(time_slice)[0][0]], s_slice, s[np.where(time_slice)[0][-1]+1:]])
  else:
    s = s_slice

  return s, len(s_slice), cadence_multiplier
