import GC_functions as func

# Maths modules:
import numpy as np
from scipy.interpolate import make_interp_spline

# Pyplot modules:
import default_setup
import matplotlib
#matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import matplotlib.patheffects as path_effects

# Astro modules:
import artpop
from astropy import units as u
from astropy.visualization import make_lupton_rgb
from astropy.io import fits

fs = 10
paths = [path_effects.Stroke(linewidth=2, foreground='k'), path_effects.Normal()]
rng = 100

# ArtPop imager:
#------------------------------------------------------------
imager = artpop.ArtImager(
  phot_system = 'HST_ACSWF', # photometric system
  diameter = 2.4 * u.m,      # effective aperture diameter
  read_noise = 3,            # read noise in electrons
  random_state = rng)
#------------------------------------------------------------

# Load bands and psfs:
#------------------------------------------------------------
bands = ['ACS_WFC_F814W', 'ACS_WFC_F606W', 'ACS_WFC_F475W']
scale = dict(ACS_WFC_F814W=1, ACS_WFC_F606W=1, ACS_WFC_F475W=1)
exptime = dict(ACS_WFC_F814W=12830 * u.s, ACS_WFC_F606W=12830 * u.s, ACS_WFC_F475W=12830 * u.s)
psf = {}
for b in bands:
  psf[b] = fits.getdata(f'./files/{b}.fits', ignore_missing_end=True)

# Scaling fractions that don't break the psf alignments, and the associated stretch values:
psf_fractions = np.array([1.0,2.0,2.6,3.1,3.8,4.1,4.6,4.9,5.8,6.2,6.9,7.2,8.6,10.1,13.,16.1,24.])
img_stretches = 10**(np.log10(psf_fractions)*2.15266 - 0.69897)
#------------------------------------------------------------

# Functions:
#============================================================

#------------------------------------------------------------
def metadata(s):
  '''
  Create a metadata string
  '''
  string = r'Time$ = %.2f\,{\rm Gyr}$' % (s['age'] / 1e3) + '\n' + \
           r'Mass$ = %s\,$M$_{\odot}$' % func.latex_float(s['mass'][s['nbound']].sum()) + '\n' + \
           r'$R_{\rm half} = %.2f\,{\rm pc}$' % s['hlr']
  return string
#------------------------------------------------------------

#------------------------------------------------------------
def orbit_trace(s, i, ax, params, trace_res=250):
  '''
  Track the past orbit of the SC for 50 Myr.
  Add a faded line to the plot which shows this trace.
  Calculate the angle of the tangent to the trace at t0,
  then apply a rotated rectangle patch.
  '''

  # Find the intended length of the tracer:
  trace_length = 50. # [Myr]
  j = i
  trace = []
  while (s[j]['age'] > s[i]['age']-trace_length) & (j>=0):
    trace.append(s[j]['rdens'] + s[j]['rg']*1e3)
    j -= 1
  trace = np.vstack(trace)[::-1]

  # Rotate the trace to match orbital frame:
  trace = np.dot(trace, params['rotation'].T)

  # Upscale the trace resolution:
  k = max(0, min(3, len(trace)-1))
  #orig_res = np.linspace(s[j+1]['age'], s[i]['age'], len(trace))
  orig_res = np.array([s[k]['age'] for k in np.arange(j+1, i+1)])
  start_time = max(s[0]['age'], s[i]['age']-trace_length)
  trace_res_mod = trace_res - int(trace_res * (start_time - (s[i]['age']-trace_length)) / trace_length)
  spline_res = np.linspace(start_time, s[i]['age'], trace_res_mod) if i else [s[i]['age']]
  trace = make_interp_spline(orig_res, trace, k=k)(spline_res)

  # Make a pyplot line for the trace:
  line_colour = (np.ones([trace_res, 4]) * np.vstack(np.arange(trace_res)))[-len(trace):] / trace_res
  line_colour[:,:3] = 1.
  trace_line = LineCollection([np.column_stack([[trace[j,0], trace[j+1,0]], [trace[j,1], trace[j+1,1]]]) for j in range(len(trace)-1)], \
                              linewidths=1, capstyle='butt', color=line_colour, joinstyle='round', zorder=99)
  ax.add_collection(trace_line)

  # Find the angle of the tangent to the trace:
  if i > 0:
    vec = np.array([trace[-1][0] - trace[-2][0], trace[-1][1] - trace[-2][1]])
    theta_xy = func.angle(vec, [1.,0.]) * np.sign(-vec[1])
  else:
    theta_xy = 0

  # Make a pyplot box for the trace:
  square = patches.Rectangle((trace[-1][0] - params['GC_width'], trace[-1][1] - params['GC_width']), \
                             width=params['GC_width']*2, height=params['GC_width']*2, \
                             facecolor='None', edgecolor='w', lw=1, ls='--', zorder=100)
  square_rotate = mpl.transforms.Affine2D().rotate_deg_around(\
                  trace[-1][0], trace[-1][1], -theta_xy * 180./np.pi) + ax.transData
  square.set_transform(square_rotate)
  ax.add_patch(square)

  return trace, theta_xy
#------------------------------------------------------------

#------------------------------------------------------------
def distance_bar(ax, panel, params):
  '''
  Create distance bars.
  '''
  for lw, color, order, capstyle in zip([3,1], ['k', 'w'], [100, 101], ['projecting', 'butt']):
    _, _, cap = ax.errorbar([params['%s_corner1' % panel], params['%s_corner1' % panel]+params['%s_ruler' % panel]], \
                             np.ones(2)*params['%s_corner2' % panel], yerr=np.ones(2)*params['%s_cap' % panel], \
                             color=color, linewidth=lw, ecolor=color, elinewidth=lw, zorder=order)
    cap[0].set_capstyle(capstyle)

  # Distance bar labels:
  ax.text(params['%s_corner1' % panel] + params['%s_ruler' % panel]/2., \
          params['%s_corner2' % panel] - 0.025*params['%s_width' % panel], \
          r'$%.0f\,$kpc' % params['%s_ruler' % panel], \
          va='top', ha='center', color='w', fontsize=fs-2, path_effects=paths)
  return
#------------------------------------------------------------
