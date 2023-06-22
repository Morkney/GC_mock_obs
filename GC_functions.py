import numpy as np

def R_half(s, type='mass', filt=[0., np.inf], bound=True):
  r = s['r']
  body = s['nbound'] if bound else np.ones_like(s['nbound']).astype('bool')
  BHs = s['kstara'] == (13 and 14)
  mass = s[type]
  filt = (mass <= filt[1]) & (mass >= filt[0])

  r, mass = r[~BHs & filt & body], mass[~BHs & filt & body]

  mass = mass[r.argsort()]
  r = np.sort(r)
  M_half = mass.sum() / 2.
  M_cum = np.cumsum(mass)
  return np.interp(M_half, M_cum, r)

def Rz_matrix(theta):
 return np.array([[np.cos(theta), -np.sin(theta), 0.], \
                 [np.sin(theta), np.cos(theta), 0.], \
                 [0., 0., 1.]])

def alignment(s, cen):
  pos_G = -(s['rg'] + cen/1e3) * 1e3
  h = np.linalg.norm(pos_G[[0,1]])
  theta_G = np.sign(pos_G[0]) * np.arcsin(pos_G[1] / h)
  if np.sign(pos_G[0]) == 1: theta_G += np.pi
  Rz = Rz_matrix(theta_G)
  s['pos'] = np.dot(s['pos'], Rz)
  return

from scipy.ndimage import gaussian_filter
from scipy.interpolate import interpn
def relief(pos, point_size, width):
  N_bins = 101
  bins = np.linspace(-width/2., width/2., N_bins)
  count, x_e, y_e = np.histogram2d(pos[:,0], pos[:,1], bins=bins)
  count /= (x_e[1] - x_e[0]) * (y_e[1] - y_e[0]) # Convert to count per pc
  count = gaussian_filter(count, sigma=0.5) # Add some smoothing

  Z = interpn(((x_e[1:] + x_e[:-1])/2., (y_e[1:]+y_e[:-1])/2.), count, \
              np.vstack([pos[:,0],pos[:,1]]).T, method="linear", bounds_error=False)
  idx = Z.argsort()

  x, y, Z, point_size = pos[:,0][idx], pos[:,1][idx], np.log10(Z[idx]), point_size[idx]
  Z[Z != Z] = np.nanmin(Z)
  return x, y, Z, point_size
