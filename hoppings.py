import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams['font.family'] = 'serif'
from numpy.linalg import eigh, inv
from scipy.optimize import curve_fit

#hopping integrals (NN, NNN, NNNN)
tt = np.array([-2.97, -0.073, -0.33])

#NN vectors
d0 = [[1/2, np.sqrt(3)/2],
    [1/2, -np.sqrt(3)/2],
    [-1,0]]

#NNN vectors
d1 = [[0, np.sqrt(3)],
    [3/2, np.sqrt(3)/2],
    [3/2, -np.sqrt(3)/2],
    [0, -np.sqrt(3)],
    [-3/2, -np.sqrt(3)/2],
    [-3/2, np.sqrt(3)/2]]

#NNNN vectors
d2 = [[-1,np.sqrt(3)],
    [-1,-np.sqrt(3)],
    [2,0]]

ll = np.array([np.hypot(d0[0][0], d0[0][1]),
    np.hypot(d1[0][0], d1[0][1]),  np.hypot(d2[0][0], d2[0][1])])

def gauss(x,t0,g):
    return t0*np.exp(g/2*x**2)

popt, covm = curve_fit(gauss, ll, -tt, p0=[7,-1.9])

xx = np.linspace(0.8,2.8,1000)
plt.plot(xx, gauss(xx,*popt), 'r-')
plt.plot(ll, -tt, 'k.')
plt.show()
