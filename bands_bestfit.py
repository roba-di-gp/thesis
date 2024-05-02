import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams['font.family'] = 'serif'
from scipy.optimize import minimize
import os
os.chdir('c:/users/gugli/desktop/tesi/codice')
from functions import *

#lattice parameter/lattice step conversion factor/2pi
conv = 2*np.pi/np.sqrt(3)
#best guess for the parameters
#energy of the isolated 2p_z electron
epz = -4.2432
#hopping integrals (NN, NNN, NNNN)
tt = [-2.97, -0.073, -0.33]
#overlap integrals
ss = [0.073, 0.018, 0.026]

#import data
bands_data = [0,0]
k_data = np.load(r'c:/users/gugli/desktop/tesi/data/MGKM_sorted.npy')
bands_data[0] = np.load(r'c:/users/gugli/desktop/tesi/data/e1_sorted.npy')
bands_data[1] = np.load(r'c:/users/gugli/desktop/tesi/data/e0_sorted.npy')

#convert from MGKM to GKMG
k_data2 = []
bands_data2 = [[],[]]
for i in range(51, 166):
    k_data2.append(k_data[i]-0.5774)
for i in range(0, 51):
    k_data2.append(k_data[i]+1)
k_data = np.array(k_data2)
for i in range(2):
    for j in range(51, 166):
        bands_data2[i].append(bands_data[i][j])
    for j in range(0, 51):
        bands_data2[i].append(bands_data[i][j])
    bands_data2[i] = np.array(bands_data2[i])
bands_data = np.array(bands_data2)

#number of points in the contour
times = 1
lenMG = 51*times
lenGK = 70*times
lenKM = 45*times
#high symmetry contour
kxMG = np.linspace(2*np.pi/3, 0, lenMG)
kyMG = np.zeros(lenMG)
kxGK = np.linspace(0, 2*np.pi/3, lenGK)
kyGK = 1/np.sqrt(3)*kxGK
kyKM = np.linspace(2*np.pi/3/np.sqrt(3),0, lenKM)
kxKM = 2*np.pi/3*np.ones(lenKM)

kindGK = np.sqrt(kxGK**2+kyGK**2)/conv
kindKM = kindGK[-1] + np.flip(kyKM)/conv
kindMG = kindKM[-1] + np.flip(kxMG)/conv

kkx = np.concatenate((kxGK, kxKM, kxMG))
kky = np.concatenate((kyGK, kyKM, kyMG))

kind = np.concatenate((kindGK, kindKM, kindMG))

for i in range(len(kind)):
    kind[i] = round(kind[i], 4)

kk = np.array([kkx, kky])
kk = np.transpose(kk)

ticks = [0, 0.6666, 1., 1.5774]
ticklabels = ['$\Gamma$','K', 'M', '$\Gamma$']


#define and minimize a chi^2
def chi(pars):
    chi = 0
    for i in range(len(kind)):
        fk, null, null = EUP(kk[i], pars)
        res0 = (bands_data[0][i]-fk[0])**2
        res1 = (bands_data[1][i]-fk[1])**2
        chi += res0+res1
    return chi

# popt = minimize(chi, [-4.2432,-2.97, -0.073, -0.33, 0.073, 0.018, 0.026], method='Nelder-Mead')
# print(popt.message)
# pp = popt.x
# np.save(r'c:/users/gugli/desktop/tesi/data/bande_bestfit.npy', pp)

pp = np.load('c:/users/gugli/desktop/tesi/data/bande_bestfit.npy')

#plot bands
my_dpi = 96
fig1, ax1 = plt.subplots(figsize=(300*(1+np.sqrt(5))/2/my_dpi, 300/my_dpi), dpi=my_dpi)
my_lw = 1.2


my_ms = 3.
dec = 3
plt.plot(k_data[::dec], bands_data[0][::dec], 'rh-', markersize = my_ms, zorder = 4, alpha = 0.5, linewidth=my_lw, label ='DFT')
plt.plot(k_data[::dec], bands_data[1][::dec], 'rh-', markersize = my_ms, zorder = 4, alpha = 0.5, linewidth=my_lw)

#NNNN
#calculate the dispersion on the high-symmetry contour
E = []
for i in range(len(kk)):
    e, u, p = EUP(kk[i], pp)
    E.append(e)

E = np.array(E)

plt.plot(kind, E[:,0], 'k-', linewidth=my_lw, zorder=4, label = 'Best fit')
plt.plot(kind, E[:,1], 'k-', linewidth=my_lw, zorder=4)

#figure details
plt.xticks(ticks, ticklabels)
plt.ylabel('Energy  [eV]')
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.15, top = 0.85)
#plt.legend(shadow=True, loc = 'upper center', prop={'size': 8})
plt.grid(axis = 'x', linestyle = '--', alpha = 0.6, zorder = -2)
plt.xlim(0, 1.5774)
plt.legend(shadow=True, loc = (0.4,0.75), prop={'size': 8})

plt.savefig('c:/users/gugli/desktop/tesi/figure/pi_bands_DFT.jpeg', dpi = my_dpi*5)

plt.show()