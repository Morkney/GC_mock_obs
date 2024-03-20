import numpy as np
import pickle
import h5py

from scipy.stats import binned_statistic_2d
from scipy.ndimage import uniform_filter
from scipy.ndimage import gaussian_filter

def normalise(a, normmin, normmax, vmin, vmax):
  normed = (a-vmin) / (vmax-vmin) * (normmax-normmin) + normmin
  normed[normed > normmax] = normmax
  normed[normed < normmin] = normmin
  return normed

def NormaliseData(data):
  return (data - np.min(data)) / (np.max(data) - np.min(data))

def nan_smooth(U, sigma=1):
  V = U.copy()
  V[np.isnan(U)] = 0
  VV = gaussian_filter(V, sigma=1)
  W = 0*U.copy()+1
  W[np.isnan(U)] = 0
  WW = gaussian_filter(W, sigma=1)
  Z = VV/WW
  Z[np.isnan(U)] = np.nan
  return Z

def streams(x, y, vx, vy, c, x_range, y_range, N_bins=40, c_type='mean', sigma=1, type='median'):

  # Setup bins:
  x_bins = np.linspace(*x_range, N_bins)
  x_points = (x_bins[1:] + x_bins[:-1]) / 2.
  y_bins = np.linspace(*y_range, N_bins)
  y_points = (y_bins[1:] + y_bins[:-1]) / 2.
  bins = [x_bins, y_bins]

  nans = ~np.isnan(vx) * ~np.isinf(vx) * ~np.isnan(vy) * ~np.isinf(vy)

  # Create colour array:
  if c_type == 'mean':
    cnorm = np.histogram2d(x[nans], y[nans], bins=bins)[0]
    colour = np.histogram2d(x[nans], y[nans], weights=c[nans], bins=bins)[0] / cnorm
    colour[np.isnan(colour) | np.isinf(colour)] = colour[~np.isnan(colour) * ~np.isinf(colour)].min()
  elif c_type == 'STD':
    vx_std = binned_statistic_2d(x[nans], y[nans], vx[nans], bins=bins, statistic='std')[0]
    vy_std = binned_statistic_2d(x[nans], y[nans], vy[nans], bins=bins, statistic='std')[0]
    colour = vx_std + vy_std

  # Create gradient images:
  if type=='median':
    vx = binned_statistic_2d(x[nans], y[nans], vx[nans], bins=bins, statistic='median')[0]
    vy = binned_statistic_2d(x[nans], y[nans], vy[nans], bins=bins, statistic='median')[0]
  elif type=='mean':
    vnorm = np.histogram2d(x[nans], y[nans], bins=bins)[0]
    vx = np.histogram2d(x[nans], y[nans], weights=vx[nans], bins=bins)[0] / vnorm
    vy = np.histogram2d(x[nans], y[nans], weights=vy[nans], bins=bins)[0] / vnorm
  vx = nan_smooth(vx, sigma)
  vy = nan_smooth(vy, sigma)
  X, Y = np.meshgrid(x_points, y_points)

  # Create density array:
  density = np.histogram2d(x[nans], y[nans], bins=bins)[0]**(1/10)
  density = normalise(density, 0.0, 2.75, density[density > 0].min(), density.max())

  return X,Y, vx.T,vy.T, density.T, colour.T
