import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
matplotlib.rcParams.update({'font.size': 12})
from numpy.linalg import eigh, inv
from scipy.optimize import curve_fit

#lattice parameter [angstrom]
lp = 2.466731
#lattice step [angstrom]
a = lp/np.sqrt(3)
#hopping integrals (NN, NNN, NNNN) [eV]
pp = np.load('c:/users/gugli/desktop/tesi/data/bande_bestfit.npy')

ttAB = np.array([pp[1], pp[3]])
ttAA = np.array([0, pp[2]])

#hopping distances [Angstrom]
llAB = np.array([1, 2])*a
llAA = np.array([0, np.sqrt(3)])*a

def gauss(x,t0,g):
    return t0*np.exp(g/2*x**2)

#fit for the interlattice hopping
pAB, covm = curve_fit(gauss, llAB, -ttAB, p0=[6, -7])

#fit for the intralattice hopping, using interlattice gaussian maximum
ttAA[0] = -gauss(0,*pAB)
pAA, covm = curve_fit(gauss, llAA, -ttAA, p0=[6,-1.])

gamma = np.array([pAA[1], pAB[1]])
np.save('c:/users/gugli/desktop/tesi/data/gamma.npy',gamma)
my_dpi = 96
fig1, ax1 = plt.subplots(figsize=(500/my_dpi, 300/my_dpi), dpi=my_dpi)

xx = np.linspace(-0.5,3.4*a,1000)
plt.plot(xx, gauss(xx,*pAA), 'k-', linewidth=1.7, zorder=-3)
plt.plot(xx, gauss(xx,*pAB), 'k-', linewidth=1.7, zorder=-3)
plt.plot(xx, gauss(xx,*pAA), 'b-', linewidth=1, zorder=-2)
plt.plot(xx, gauss(xx,*pAB), 'r-', linewidth=1, zorder=-2)
plt.plot(llAA[1], -ttAA[1], 'bh', markersize = 6, zorder=0, label = '$t_{AA}$')
plt.plot(llAB, -ttAB, 'rh', markersize = 6, zorder = 0, label = '$t_{AB}$')
plt.plot(llAA[1], -ttAA[1], 'kh', markersize = 8, zorder=-1)
plt.plot(llAB, -ttAB, 'kh', markersize = 8, zorder = -1)

red_patch = matplotlib.patches.Patch(facecolor='red',edgecolor='k', label='$t_{AB}$')
blue_patch = matplotlib.patches.Patch(facecolor='blue',edgecolor='k', label='$t_{AA}$')
plt.legend(handles=[blue_patch, red_patch], shadow=True, loc = 'upper right', prop={'size': 12})

#plt.legend(shadow=True, loc = 'upper right', prop={'size': 12})

plt.text(0.92*a, 4.2, '$t^{(1)}$', fontsize='medium')
plt.text(1.65*a, 1.6, '$t^{(2)}$', fontsize='medium')
plt.text(1.95*a, 2.4, '$t^{(3)}$', fontsize='medium')

plt.arrow(1.0*a, 4.06, 0., -0.5, head_width=0.06, head_length=0.18, facecolor='k', linewidth=0.6)
plt.arrow(1.73*a, 1.46, 0., -0.9, head_width=0.06, head_length=0.18, facecolor='k', linewidth=0.6)
plt.arrow(2.0*a, 2.22, 0., -1.3, head_width=0.06, head_length=0.18, facecolor='k', linewidth=0.6)

plt.xlabel(r'$r \,\,\,[\operatorname{\AA}]$')
plt.ylabel('$t(r)$  [eV]')
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.9)
#plt.grid(linestyle = '--', alpha = 0.5, zorder = -2)
plt.xlim(0.,3.4*a)
plt.savefig('c:/users/gugli/desktop/tesi/figure/hoppings.jpeg', dpi = my_dpi*5)

plt.show()
