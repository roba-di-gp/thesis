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
plt.rcParams['mathtext.bf'] = 'Palatino Linotype:bold'


#lattice parameter/lattice step conversion factor * 2pi
conv_lattice = 2*np.pi/np.sqrt(3)
#conversion from cm^-1 to meV
conv = 0.00012398*1e3
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

data = np.loadtxt("c:/users/gugli/desktop/tesi/Per_Guglielmo_2D_cutoff/graphene_phonon_disp.gnuplot")

k_data = np.unique(data[:, 0])
za = np.load('c:/users/gugli/desktop/tesi/data/ZA.npy')
zo = np.load('c:/users/gugli/desktop/tesi/data/ZO.npy')
ta = np.load('c:/users/gugli/desktop/tesi/data/TA.npy')
la = np.load('c:/users/gugli/desktop/tesi/data/LA.npy')
to = np.load('c:/users/gugli/desktop/tesi/data/TO.npy')
lo = np.load('c:/users/gugli/desktop/tesi/data/LO.npy')

lines_data = np.array([za, zo, ta, la, to, lo])*conv

#number of points in the contour
times = 1
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

width = 40
wgz= []
for i in range(len(kind)):
    wgz.append(1 - np.exp(-(i)**2/width**2) - np.exp(-(i-90)**2/width**2))
wgz = np.array(wgz)
wkz = []
for i in range(len(kind)):
    wkz.append(1 - np.exp(-(i-40)**2/width**2))
wkz = np.array(wkz)
wmz = []
for i in range(len(kind)):
    wmz.append(1 - np.exp(-(i-60)**2/width**2))
wmz = np.array(wmz)

wz = wmz*wkz

width = 10
wgxy= []
for i in range(len(kind)):
    wgxy.append(1 - np.exp(-(i)**2/width**2) - np.exp(-(i-90)**2/width**2))
wgxy = np.array(wgxy)
wkxy = []
for i in range(len(kind)):
    wkxy.append(1 - np.exp(-(i-40)**2/width**2))
wkxy = np.array(wkxy)
wmxy = []
for i in range(len(kind)):
    wmxy.append(1 - np.exp(-(i-60)**2/width**2))
wmxy = np.array(wmxy)

wxy = wmxy*wkxy

#define and minimize two independent loss functions
def loss_z(pz):
    lossz = 0
    for i in range(len(kind)):
        Ez = fun.ph_z(kk[i], pz)
        for l in range(2):
            res = wz[i]*abs(lines_data[l,i] - Ez[l])**2
            lossz += res
    return lossz

def loss_xy(pxy):
    lossxy = 0
    for i in range(len(kind)):
        Exy = fun.ph_xy(kk[i], pxy)
        for l in range(4):
            res = wxy[i]*abs(lines_data[l+2,i] - Exy[l])**2
            lossxy += res
    return lossxy

# #minimize absolute deviations
# popt_z = minimize(loss_z, pp_falkovsky[0:2], method='Nelder-Mead', tol = 1e-5)
# print(popt_z.message)
# pp_z = popt_z.x
# popt_xy = minimize(loss_xy, pp_falkovsky[2:6], method='Nelder-Mead', tol = 1e-5)
# print(popt_xy.message)
# pp_xy = popt_xy.x
# np.save(r'c:/users/gugli/desktop/tesi/data/phonons_bestfit_z.npy', pp_z)
# np.save(r'c:/users/gugli/desktop/tesi/data/phonons_bestfit_xy.npy', pp_xy)

pp_z = np.load(r'c:/users/gugli/desktop/tesi/data/phonons_bestfit_z.npy')
pp_xy = np.load(r'c:/users/gugli/desktop/tesi/data/phonons_bestfit_xy.npy')

#calculate the dispersion on the high-symmetry contour
Eout = []
Ein = []
for i in range(len(kk)):
    eout = fun.ph_z(kk[i], pp_z)
    ein = fun.ph_xy(kk[i], pp_xy)
    Eout.append(eout)
    Ein.append(ein)

#convert to numpy array (easier to plot)
Eout = np.array(Eout)
Ein = np.array(Ein)

#convert to meV
lines = np.array([Eout[:,0], Eout[:,1], Ein[:,0], Ein[:,1], Ein[:,2], Ein[:,3]])


#plot modes
my_dpi = 96
fig1, ax1 = plt.subplots(figsize=(600/my_dpi,330/my_dpi), dpi=my_dpi)

my_lw = 1.2
my_ms = 2.5

for i in range(2):
    plt.plot(kind, lines[i][:], 'k-', linewidth=my_lw, zorder=2)
    plt.plot(kind, lines_data[i,:], 'rh-', linewidth=my_lw, alpha=0.5, markersize = my_ms, zorder=1)

for i in range(2,6):
    plt.plot(kind, lines[i][:], 'k-', linewidth=my_lw, zorder=2)
    plt.plot(kind, lines_data[i,:], 'rh-', linewidth=my_lw, alpha=0.5, markersize = my_ms, zorder=1)

plt.plot(kind, lines_data[5,:], 'rh-', linewidth=my_lw, alpha=0.5, markersize = my_ms, label='DFT', zorder = 1)
plt.plot(kind, lines[5][:], 'k-', linewidth=my_lw, zorder=2, label='Best fit')

#figure details
ticks = [0, 0.6666, 1., 1.5774]
ticklabels = ['$\Gamma$','K', 'M', '$\Gamma$']
plt.xticks(ticks, ticklabels)
plt.ylabel('Phonon energy  [meV]')
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.1, top = 0.9)
plt.legend(shadow=True, loc = (0.85,0.62), prop={'size': 10})
plt.grid(axis = 'x', linestyle = '--', alpha = 0.6, zorder = -5)
my_fs = 9
#convert to meV
plt.text(4./conv_lattice, conv*1550, 'LO', fontsize=my_fs)
plt.text(1.4/conv_lattice, conv*1550, 'LO', fontsize=my_fs)
plt.text(4.9/conv_lattice, conv*1390, 'TO', fontsize=my_fs)
plt.text(0.6/conv_lattice, conv*1370, 'TO', fontsize=my_fs)
plt.text(4.5/conv_lattice, conv*1000, 'LA', fontsize=my_fs)
plt.text(1./conv_lattice, conv*1000, 'LA', fontsize=my_fs)
plt.text(1.05/conv_lattice, conv*360, 'TA', fontsize=my_fs)
plt.text(4.3/conv_lattice, conv*360, 'TA', fontsize=my_fs)
plt.text(0.1/conv_lattice, conv*940, 'ZO', fontsize=my_fs)
plt.text(5.3/conv_lattice, conv*940, 'ZO', fontsize=my_fs)
plt.text(1.2/conv_lattice, conv*110, 'ZA', fontsize=my_fs)
plt.text(4.2/conv_lattice, conv*110, 'ZA', fontsize=my_fs)

plt.xlim(0, 1.5774)
plt.ylim(0, 210)
plt.savefig('c:/users/gugli/desktop/tesi/figure/phonons_DFT.jpeg', dpi = my_dpi*5)

plt.show()
##force constants from high-symmetry points

za = np.load('c:/users/gugli/desktop/tesi/data/ZA.npy')
zo = np.load('c:/users/gugli/desktop/tesi/data/ZO.npy')
ta = np.load('c:/users/gugli/desktop/tesi/data/TA.npy')
la = np.load('c:/users/gugli/desktop/tesi/data/LA.npy')
to = np.load('c:/users/gugli/desktop/tesi/data/TO.npy')
lo = np.load('c:/users/gugli/desktop/tesi/data/LO.npy')

alpha_z = -1/6*zo[0]**2
gamma_z = -1/8*(zo[60]**2 + 3*alpha_z)
pp_z = [alpha_z, gamma_z]
pp_z = [pp_falkovsky[0], pp_falkovsky[1]]

Eout = []
Ein = []
for i in range(len(kk)):
    eout = fun.ph_z(kk[i], pp_z)
    #ein = fun.ph_xy(kk[i], pp_xy)
    Eout.append(eout)
    #Ein.append(ein)

#convert to numpy array (easier to plot)
Eout = np.array(Eout)
#Ein = np.array(Ein)

#convert to meV
lines = np.array([Eout[:,0], Eout[:,1]])
lines_data = np.array([za, zo, ta, la, to, lo])*conv
#plot modes
my_dpi = 96
fig1, ax1 = plt.subplots(figsize=(650/my_dpi, 400/my_dpi), dpi=my_dpi)

my_lw = 1.2
my_ms = 2.5

for i in range(2):
    plt.plot(kind, lines[i][:], 'k-', linewidth=my_lw, zorder=2)
    plt.plot(kind, lines_data[i,:], 'rh-', linewidth=my_lw, alpha=0.5, markersize = my_ms, zorder=1)

#figure details
ticks = [0, 0.6666, 1., 1.5774]
ticklabels = ['$\Gamma$','K', 'M', '$\Gamma$']
plt.xticks(ticks, ticklabels)
plt.ylabel('Phonon energy  [meV]')
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.9)
#plt.legend(shadow=True, loc = (0.35,0.05), prop={'size': 8})
plt.grid(axis = 'x', linestyle = '--', alpha = 0.6, zorder = -5)
my_fs = 9

plt.text(4./conv_lattice, conv*1550, 'LO', fontsize=my_fs)
plt.text(1.4/conv_lattice, conv*1550, 'LO', fontsize=my_fs)
plt.text(4.9/conv_lattice, conv*1390, 'TO', fontsize=my_fs)
plt.text(0.6/conv_lattice, conv*1370, 'TO', fontsize=my_fs)
plt.text(4.5/conv_lattice, conv*1000, 'LA', fontsize=my_fs)
plt.text(1./conv_lattice, conv*1000, 'LA', fontsize=my_fs)
plt.text(1.05/conv_lattice, conv*360, 'TA', fontsize=my_fs)
plt.text(4.3/conv_lattice, conv*360, 'TA', fontsize=my_fs)
plt.text(0.1/conv_lattice, conv*940, 'ZO', fontsize=my_fs)
plt.text(5.3/conv_lattice, conv*940, 'ZO', fontsize=my_fs)
plt.text(1.2/conv_lattice, conv*110, 'ZA', fontsize=my_fs)
plt.text(4.2/conv_lattice, conv*110, 'ZA', fontsize=my_fs)

plt.xlim(0, 1.5774)
#plt.ylim(0, 210)
plt.savefig('c:/users/gugli/desktop/tesi/figure/phonons_DFT2.jpeg', dpi = my_dpi*5)

plt.show()