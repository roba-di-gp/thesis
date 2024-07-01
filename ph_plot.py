import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from numpy.linalg import eigh, inv
from scipy.optimize import minimize
import os
os.chdir('c:/users/gugli/desktop/tesi/codice')
import functions as fun

matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
plt.rcParams['font.family'] = 'Palatino Linotype'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Palatino Linotype'
plt.rcParams['mathtext.it'] = 'Palatino Linotype:italic'
plt.rcParams['mathtext.bf'] = 'Palatino Linotype:bolditalic'


#lattice parameter/lattice step conversion factor * 2pi
conv_lattice = 2*np.pi/np.sqrt(3)
#conversion from cm^-1 to eV
conv = 0.00012398
#from cm^-2 to meV^2
conv2 = conv**2
#flexural force constants
#alpha_z, gamma_z
alpha_z = -1.176*10**5*conv2; gamma_z = 0.190*10**5*conv2

#in-plane force constants
#alpha, beta, gamma, delta
alpha = -4.046*10**5*conv2; beta = 1.107*10**5*conv2;
gamma = -0.238*10**5*conv2; delta = -1.096*10**5*conv2

pp_falkovsky = [alpha_z, gamma_z, alpha, beta, gamma, delta]

def sort(arr):
    n = len(arr)
    arr_out = np.copy(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr_out[j] > arr_out[j+1]:
                arr_out[j], arr_out[j+1] = arr_out[j+1], arr_out[j]
    return arr_out

#number of points in the contour
times = 4
lenGK = 41*times
lenKM = 21*times
lenMG = 31*times
#high symmetry contour
kxMG = np.linspace(2*np.pi/3, 0, lenMG)
kyMG = np.zeros(lenMG)
kxGK = np.linspace(0, 2*np.pi/3, lenGK)
kyGK = 1/np.sqrt(3)*kxGK
kyKM = np.linspace(2*np.pi/3/np.sqrt(3),0, lenKM)
kxKM = 2*np.pi/3*np.ones(lenKM)

kindGK = np.sqrt(kxGK**2+kyGK**2)/conv_lattice
kindKM = kindGK[-1] + np.flip(kyKM)/conv_lattice
kindMG = kindKM[-1] + np.flip(kxMG)/conv_lattice

kkx = np.concatenate((kxGK, kxKM[1:], kxMG[1:]))
kky = np.concatenate((kyGK, kyKM[1:], kyMG[1:]))

kind = np.concatenate((kindGK, kindKM[1:], kindMG[1:]))

ticks = [0, 0.6666, 1., 1.5774]
ticklabels = ['$\Gamma$','K', 'M', '$\Gamma$']

for i in range(len(kind)):
    kind[i] = round(kind[i], 4)

kk = np.array([kkx, kky])
kk = np.transpose(kk)

#calculate the dispersion on the high-symmetry contour
Eout = []
Ein = []
for i in range(len(kk)):
    eout = fun.ph_z(kk[i], pp_falkovsky[0:2])
    ein = fun.ph_xy(kk[i], pp_falkovsky[2:6])
    Eout.append(eout)
    Ein.append(ein)

#convert to numpy array (easier to plot)
Eout = np.array(Eout)
Ein = np.array(Ein)

#convert to meV
lines = np.array([Eout[:,0], Eout[:,1], Ein[:,0], Ein[:,1], Ein[:,2], Ein[:,3]])*1e3
#plot modes
my_dpi = 96
fig1, ax1 = plt.subplots(figsize=(1.8*350/my_dpi, 350/my_dpi), dpi=my_dpi)

my_lw = 1.2
my_ms = 2.5

for i in range(2):
    plt.plot(kind, lines[i][:], 'b-', linewidth=my_lw, zorder=2)

for i in range(2,6):
    plt.plot(kind, lines[i][:], 'k-', linewidth=my_lw, zorder=2)

#plt.plot(kind, lines[4][:], 'b-', linewidth=my_lw, zorder=2)

#figure details
ticks = [0, 0.6666, 1., 1.5774]
ticklabels = ['$\Gamma$','K', 'M', '$\Gamma$']
plt.xticks(ticks, ticklabels)
plt.ylabel('Phonon energy  [meV]')
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.1, top = 0.9)
#plt.legend(shadow=True, loc = (0.35,0.05), prop={'size': 8})
plt.grid(axis = 'x', linestyle = '--', alpha = 0.6, zorder = -5)
my_fs = 9
#convert to meV
conv = conv*1e3
plt.text(4./conv_lattice, conv*1550, 'LO', fontsize=my_fs)
plt.text(1.4/conv_lattice, conv*1550, 'LO', fontsize=my_fs)
plt.text(4.9/conv_lattice, conv*1390, 'TO', fontsize=my_fs)
plt.text(0.6/conv_lattice, conv*1370, 'TO', fontsize=my_fs)
plt.text(4.5/conv_lattice, conv*1000, 'LA', fontsize=my_fs)
plt.text(1./conv_lattice, conv*1000, 'LA', fontsize=my_fs)
plt.text(1.05/conv_lattice, conv*300, 'TA', fontsize=my_fs)
plt.text(4.3/conv_lattice, conv*300, 'TA', fontsize=my_fs)
plt.text(0.1/conv_lattice, conv*880, 'ZO',color = 'b', fontsize=my_fs)
plt.text(5.3/conv_lattice, conv*880, 'ZO',color = 'b', fontsize=my_fs)
plt.text(1.2/conv_lattice, conv*70, 'ZA', color = 'b',fontsize=my_fs)
plt.text(4.2/conv_lattice, conv*70, 'ZA', color = 'b',fontsize=my_fs)

plt.xlim(0, 1.5774)
plt.ylim(0, 210)
plt.savefig('c:/users/gugli/desktop/tesi/figure/phonons_falkovsky.png', dpi = my_dpi*5)

plt.show()