import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams['font.family'] = 'serif'
from numpy.linalg import eigh, inv
from scipy.optimize import minimize
import os
os.chdir('c:/users/gugli/desktop/tesi/codice')
import functions as fun

#lattice parameter/lattice step conversion factor * 2pi
conv_lattice = 2*np.pi/np.sqrt(3)
#conversion from cm^-1 to eV
conv = 0.00012398
#from cm^-2 to meV^2
conv2 = conv**2
ncont = 277
nfbz = 48
#gap [meV]
gap = 1

pp_ph_z = np.load(r'c:/users/gugli/desktop/tesi/data/phonons_bestfit_z.npy')
pp_ph_xy = np.load(r'c:/users/gugli/desktop/tesi/data/phonons_bestfit_xy.npy')
#geometric term in the dynamical matrix [eV^2]
Dgeo = np.load('c:/users/gugli/desktop/tesi/data/Dgeo_'+str(ncont)+'_'+str(nfbz)+'_'+str(gap)+'mev.npy')
#to meV^2
Dgeo = Dgeo*1e6

qq = np.load('c:/users/gugli/desktop/tesi/data/ph_contour_vecs_'+str(ncont)+'.npy')
qind = np.load('c:/users/gugli/desktop/tesi/data/ph_contour_ind_'+str(ncont)+'.npy')

ticks = [0, 0.6666, 1., 1.5774]
ticklabels = ['$\Gamma$','K', 'M', '$\Gamma$']

#calculate the dispersion on the high-symmetry contour
Eout = []
Ein = []
for i in range(len(qq)):
    eout = fun.ph_z(qq[i], pp_ph_z)
    ein = fun.ph_xy(qq[i], pp_ph_xy)
    Eout.append(eout)
    Ein.append(ein)

Ein_ng = []
for i in range(len(qq)):
    ein_ng = fun.ph_xy_NG(qq[i], pp_ph_xy, Dgeo[i])
    Ein_ng.append(ein_ng)


#convert to numpy array (easier to plot)
Eout = np.array(Eout)
Ein = np.array(Ein)
Ein_ng = np.array(Ein_ng)

#convert to meV
lines = np.array([Eout[:,0], Eout[:,1], Ein[:,0], Ein[:,1], Ein[:,2], Ein[:,3]])
lines_ng = np.array([Ein_ng[:,0],Ein_ng[:,1],Ein_ng[:,2],Ein_ng[:,3]])

#plot modes
my_dpi = 96
fig1, ax1 = plt.subplots(1, figsize=(650/my_dpi, 400/my_dpi), dpi=my_dpi)

my_lw = 1.2
my_ms = 1.

for i in range(2,5):
    plt.plot(qind, lines[i,:], 'k-', linewidth=my_lw, markersize = my_ms,zorder=2)
for i in range(4):
    plt.plot(qind[1:-1], lines_ng[i,1:-1], 'b.', linewidth=my_lw,markersize = my_ms, zorder=2)

plt.plot(qind, lines[5][:], 'k-', linewidth=my_lw, zorder=2,markersize = my_ms, label='With geometry')
plt.plot(qind, lines_ng[3][:], 'b.', linewidth=my_lw, zorder=2,markersize = my_ms, label='Without geometry')

#figure details
ticks = [0, 0.6666, 1., 1.5774]
ticklabels = ['$\Gamma$','K', 'M', '$\Gamma$']
plt.xticks(ticks, ticklabels)
plt.ylabel('Phonon energy  [meV]')
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.9)
plt.legend(shadow=True, loc = (0.35,0.05), prop={'size': 8})
plt.grid(axis = 'x', linestyle = '--', alpha = 0.6, zorder = -5)
my_fs = 12
#convert to meV
conv = conv*1e3
plt.text(4./conv_lattice, conv*1600, 'LO', fontsize=my_fs)
plt.text(1.4/conv_lattice, conv*1600, 'LO', fontsize=my_fs)
plt.text(4.9/conv_lattice, conv*1390, 'TO', fontsize=my_fs)
plt.text(0.6/conv_lattice, conv*1370, 'TO', fontsize=my_fs)
plt.text(4.6/conv_lattice, conv*1000, 'LA', fontsize=my_fs)
plt.text(0.9/conv_lattice, conv*1000, 'LA', fontsize=my_fs)
plt.text(1.05/conv_lattice, conv*300, 'TA', fontsize=my_fs)
plt.text(4.3/conv_lattice, conv*300, 'TA', fontsize=my_fs)
plt.title('$\Delta$ = %.0f meV'%gap)

plt.xlim(0, 1.5774)
#plt.ylim(0, 210)
plt.savefig('c:/users/gugli/desktop/tesi/figure/phonons_ng_'+str(ncont)+'_'+str(nfbz)+'_'+str(gap) +'mev.png', dpi = my_dpi*5)

plt.show()

##Gamma point neighbors (right), optical ph

#lattice parameter/lattice step conversion factor * 2pi
conv_lattice = 2*np.pi/np.sqrt(3)
#conversion from cm^-1 to eV
conv = 0.00012398
#from cm^-2 to meV^2
conv2 = conv**2


pp_ph = np.load(r'c:/users/gugli/desktop/tesi/data/phonons_bestfit.npy')
#geometric term in the dynamical matrix [eV^2]
Dgeo = np.load('c:/users/gugli/desktop/tesi/data/Dgeo_GK.npy')
Dgeo = Dgeo*1e6
qq = np.load('c:/users/gugli/desktop/tesi/data/qGK.npy')
qind = np.load('c:/users/gugli/desktop/tesi/data/qGKind.npy')

#calculate the dispersion on the high-symmetry contour
Eout = []
Ein = []
for i in range(len(qq)):
    eout = fun.ph_z(qq[i], pp_ph)
    ein = fun.ph_xy(qq[i], pp_ph)
    Eout.append(eout)
    Ein.append(ein)

Ein_ng = []
for i in range(len(qq)):
    ein_ng = fun.ph_xy_NG(qq[i], pp_ph, Dgeo[i])
    Ein_ng.append(ein_ng)

#convert to numpy array (easier to plot)
Eout = np.array(Eout)
Ein = np.array(Ein)
Ein_ng = np.array(Ein_ng)

#convert to meV
lines = np.array([Ein[:,0], Ein[:,1]])
lines_ng = np.array([Ein_ng[:,0],Ein_ng[:,1]])

#plot modes
my_dpi = 96
fig1, ax1 = plt.subplots(1, figsize=(650/my_dpi, 400/my_dpi), dpi=my_dpi)

my_lw = 1.2
my_ms = 2.5


plt.plot(qind, lines[0,:], 'k-', linewidth=my_lw, zorder=2)
plt.plot(qind, lines_ng[0,:], 'bh-', linewidth=my_lw, markersize = 4,zorder=2)
plt.plot(qind, lines[1][:], 'k-', linewidth=my_lw, zorder=2, label='With geometry')
plt.plot(qind, lines_ng[1][:], 'bh-', linewidth=my_lw, zorder=2, markersize = 4, label='Without geometry')

#figure details
ticks = [0, qind[-1]]
ticklabels = ['$\Gamma$', '$\Gamma + (%.2f, %.2f)$'%(qq[-1,0], qq[-1,1])]
plt.xticks(ticks, ticklabels)
plt.ylabel('Phonon energy  [meV]')
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.9)
plt.legend(shadow=True, loc = (0.35,0.4), prop={'size': 12})
plt.grid(axis = 'x', linestyle = '--', alpha = 0.6, zorder = -5)
my_fs = 12
#convert to meV
conv = conv*1e3
plt.text(0.003, 194.3, 'LO $\equiv$ TO', fontsize=my_fs)
plt.text(0.002, 198.6, 'LO', fontsize=my_fs, color = 'b')
plt.text(0.0012, 195.5, 'TO', fontsize=my_fs, color = 'b')

plt.xlim(0, qind[-1])
#plt.ylim(0, 210)
plt.savefig('c:/users/gugli/desktop/tesi/figure/phonons_ng_GK.png', dpi = my_dpi*5)

plt.show()

##Gamma point neighbors (left), optical ph

#lattice parameter/lattice step conversion factor * 2pi
conv_lattice = 2*np.pi/np.sqrt(3)
#conversion from cm^-1 to eV
conv = 0.00012398
#from cm^-2 to meV^2
conv2 = conv**2

pp_ph = np.load(r'c:/users/gugli/desktop/tesi/data/phonons_bestfit.npy')
#geometric term in the dynamical matrix [eV^2]
Dgeo = np.load('c:/users/gugli/desktop/tesi/data/Dgeo_MG.npy')
qq = np.load('c:/users/gugli/desktop/tesi/data/qMG.npy')
qind = np.load('c:/users/gugli/desktop/tesi/data/qMGind.npy')

#calculate the dispersion on the high-symmetry contour
Eout = []
Ein = []
for i in range(len(qq)):
    eout = fun.ph_z(qq[i], pp_ph)
    ein = fun.ph_xy(qq[i], pp_ph)
    Eout.append(eout)
    Ein.append(ein)

Ein_ng = []
for i in range(len(qq)):
    ein_ng = fun.ph_xy_NG(qq[i], pp_ph, Dgeo[i])
    Ein_ng.append(ein_ng)

#convert to numpy array (easier to plot)
Eout = np.array(Eout)
Ein = np.array(Ein)
Ein_ng = np.array(Ein_ng)

#convert to meV
lines = np.array([Ein[:,2], Ein[:,3]])*1e3
lines_ng = np.array([Ein_ng[:,2],Ein_ng[:,3]])*1e3

#plot modes
my_dpi = 96
fig1, ax1 = plt.subplots(1, figsize=(650/my_dpi, 400/my_dpi), dpi=my_dpi)

my_lw = 1.2
my_ms = 2.5


plt.plot(qind, lines[0,:], 'k-', linewidth=my_lw, zorder=2)
plt.plot(qind, lines_ng[0,:], 'bh-', linewidth=my_lw, markersize = 4,zorder=2)
plt.plot(qind, lines[1][:], 'k-', linewidth=my_lw, zorder=2, label='With geometry')
plt.plot(qind, lines_ng[1][:], 'bh-', linewidth=my_lw, zorder=2, markersize = 4, label='Without geometry')

#figure details
ticks = [qind[0], qind[-1]]
ticklabels = ['$\Gamma - (%.2f, %.2f)$'%(qq[0,0], qq[0,1]), '$\Gamma$']
plt.xticks(ticks, ticklabels)
plt.ylabel('Phonon energy  [meV]')
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.9)
plt.legend(shadow=True, loc = (0.35,0.7), prop={'size': 12})
plt.grid(axis = 'x', linestyle = '--', alpha = 0.6, zorder = -5)
my_fs = 12
#convert to meV
conv = conv*1e3
plt.text(1.574, 194.3, 'LO $\equiv$ TO', fontsize=my_fs)
plt.text(1.575, 198.6, 'LO', fontsize=my_fs, color = 'b')
plt.text(1.576, 195.8, 'TO', fontsize=my_fs, color = 'b')

plt.xlim(qind[0], qind[-1])
#plt.ylim(0, 210)
plt.savefig('c:/users/gugli/desktop/tesi/figure/phonons_ng_MG.png', dpi = my_dpi*5)

plt.show()

##M point neighbors, optical ph

#lattice parameter/lattice step conversion factor * 2pi
conv_lattice = 2*np.pi/np.sqrt(3)
#conversion from cm^-1 to eV
conv = 0.00012398
#from cm^-2 to meV^2
conv2 = conv**2

#geometric term in the dynamical matrix [eV^2]
Dgeo = np.load('c:/users/gugli/desktop/tesi/data/Dgeo_M.npy')
qq = np.load('c:/users/gugli/desktop/tesi/data/qM.npy')
qind = np.load('c:/users/gugli/desktop/tesi/data/qMind.npy')

#calculate the dispersion on the high-symmetry contour
Eout = []
Ein = []
for i in range(len(qq)):
    ein = fun.ph_xy(qq[i], pp_ph_xy)
    Eout.append(eout)
    Ein.append(ein)

Ein_ng = []
for i in range(len(qq)):
    ein_ng = fun.ph_xy_NG(qq[i], pp_ph_xy, Dgeo[i])
    Ein_ng.append(ein_ng)

#convert to numpy array (easier to plot)
Eout = np.array(Eout)
Ein = np.array(Ein)
Ein_ng = np.array(Ein_ng)

#convert to meV
Ein =  np.transpose(Ein)*1e3
Ein_ng =  np.transpose(Ein_ng)*1e3

#plot modes
my_dpi = 96
fig1, ax1 = plt.subplots(1, figsize=(650/my_dpi, 400/my_dpi), dpi=my_dpi)

my_lw = 1.2
my_ms = 2.5

plt.plot(qind, Ein[2], 'k-', linewidth=my_lw, zorder=2)
plt.plot(qind, Ein_ng[2], 'bh-', linewidth=my_lw, markersize = 4,zorder=2)
plt.plot(qind,  Ein[3], 'k-', linewidth=my_lw, zorder=2, label='With geometry')
plt.plot(qind,  Ein_ng[3], 'bh-', linewidth=my_lw, zorder=2, markersize = 4, label='Without geometry')

#figure details
ticks = [qind[0], 1, qind[-1]]
ticklabels = ['$M - (%.2f, %.2f)$'%(qq[34,0]-qq[0,0], qq[0,1]),'$M$',  '$M + (%.2f, %.2f)$'%(qq[34,0]-qq[-1,0], qq[-1,1])]
plt.xticks(ticks, ticklabels)
plt.ylabel('Phonon energy  [meV]')
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.9)
plt.legend(shadow=True, loc = (0.35,0.5), prop={'size': 12})
plt.grid(axis = 'x', linestyle = '--', alpha = 0.6, zorder = -5)
my_fs = 12
#convert to meV
conv = conv*1e3
plt.text(1.003, 175.3, 'LO', fontsize=my_fs)
plt.text(1.003, 157.5, 'TO', fontsize=my_fs)
plt.text(1.003, 178.7, 'LO', fontsize=my_fs, color = 'b')
plt.text(1.003, 160.3, 'TO', fontsize=my_fs, color = 'b')

plt.xlim(qind[0], qind[-1])
#plt.ylim(156, 181)
plt.savefig('c:/users/gugli/desktop/tesi/figure/phonons_ng_M.png', dpi = my_dpi*5)

plt.show()

##K point neighbors

#lattice parameter/lattice step conversion factor * 2pi
conv_lattice = 2*np.pi/np.sqrt(3)
#conversion from cm^-1 to eV
conv = 0.00012398
#from cm^-2 to meV^2
conv2 = conv**2

pp_ph_z = np.load(r'c:/users/gugli/desktop/tesi/data/phonons_bestfit_z.npy')
pp_ph_xy = np.load(r'c:/users/gugli/desktop/tesi/data/phonons_bestfit_xy.npy')
#geometric term in the dynamical matrix [eV^2]
Dgeo = np.load('c:/users/gugli/desktop/tesi/data/Dgeo_K.npy')
qq = np.load('c:/users/gugli/desktop/tesi/data/qK.npy')
qind = np.load('c:/users/gugli/desktop/tesi/data/qKind.npy')

#calculate the dispersion on the high-symmetry contour
Eout = []
Ein = []
for i in range(len(qq)):
    eout = fun.ph_z(qq[i], pp_ph_z)
    ein = fun.ph_xy(qq[i], pp_ph_xy)
    Eout.append(eout)
    Ein.append(ein)

Ein_ng = []
for i in range(len(qq)):
    ein_ng = fun.ph_xy_NG(qq[i], pp_ph_xy, Dgeo[i])
    Ein_ng.append(ein_ng)

#convert to numpy array (easier to plot)
Eout = np.array(Eout)
Ein = np.array(Ein)
Ein_ng = np.array(Ein_ng)

#convert to meV
Ein =  np.transpose(Ein)*1e3
Ein_ng =  np.transpose(Ein_ng)*1e3

#plot modes
my_dpi = 96
fig1, ax1 = plt.subplots(1, figsize=(650/my_dpi, 400/my_dpi), dpi=my_dpi)

my_lw = 1.2
my_ms = 3

plt.plot(qind, Ein[2], 'k-', linewidth=my_lw, zorder=2)
plt.plot(qind, Ein_ng[2], 'bh-', linewidth=my_lw, markersize = my_ms,zorder=2)
plt.plot(qind, Ein[0], 'k-', linewidth=my_lw, zorder=2)
plt.plot(qind, Ein_ng[0], 'bh-', linewidth=my_lw, markersize = my_ms,zorder=2)
plt.plot(qind, Ein[1], 'k-', linewidth=my_lw, zorder=2)
plt.plot(qind, Ein_ng[1], 'bh-', linewidth=my_lw, markersize = my_ms,zorder=2)
plt.plot(qind,  Ein[3], 'k-', linewidth=my_lw, zorder=2, label='With geometry')
plt.plot(qind,  Ein_ng[3], 'bh-', linewidth=my_lw, zorder=2, markersize = my_ms, label='Without geometry')

#figure details
ticks = [qind[0], 2/3, qind[-1]]
ticklabels = ['$K - (%.2f, %.2f)$'%(qq[34,0]-qq[0,0],qq[34,1]- qq[0,1]),'$K$',  '$K + (%.2f, %.2f)$'%(qq[34,0]-qq[-1,0], qq[34,1]-qq[-1,1])]
plt.xticks(ticks, ticklabels)
plt.ylabel('Phonon energy  [meV]')
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.9)
plt.legend(shadow=True, loc = (0.65,0.52), prop={'size': 10})
plt.grid(axis = 'x', linestyle = '--', alpha = 0.6, zorder = -5)
my_fs = 12
#convert to meV
conv = conv*1e3
plt.text(0.67, 168, 'LO', fontsize=my_fs)
plt.text(0.665, 149, 'TO $\equiv$ LA', fontsize=my_fs)
plt.text(0.67, 135.5, 'TA', fontsize=my_fs)
plt.text(0.665, 172, 'LO', fontsize=my_fs, color = 'b')
plt.text(0.664, 162, 'TO', fontsize=my_fs, color = 'b')
plt.text(0.664, 154.5, 'LA', fontsize=my_fs, color = 'b')
plt.text(0.6645, 138.5, 'TA', fontsize=my_fs, color = 'b')
plt.title('$\Delta$ = 10 meV')

plt.xlim(qind[0], qind[-1])
#plt.ylim(156, 181)
plt.savefig('c:/users/gugli/desktop/tesi/figure/phonons_ng_K.png', dpi = my_dpi*5)

plt.show()

##Dgeo eigenvalues

ncont = 277
nfbz = 48

pp_ph_z = np.load(r'c:/users/gugli/desktop/tesi/data/phonons_bestfit_z.npy')
pp_ph_xy = np.load(r'c:/users/gugli/desktop/tesi/data/phonons_bestfit_xy.npy')
#geometric term in the dynamical matrix [eV^2]
Dgeo = np.load('c:/users/gugli/desktop/tesi/data/Dgeo_'+str(ncont)+'_'+str(nfbz)+'.npy')
#to meV^2
Dgeo = Dgeo*1e6

qq = np.load('c:/users/gugli/desktop/tesi/data/ph_contour_vecs_'+str(ncont)+'.npy')
qind = np.load('c:/users/gugli/desktop/tesi/data/ph_contour_ind_'+str(ncont)+'.npy')

ticks = [0, 0.6666, 1., 1.5774]
ticklabels = ['$\Gamma$','K', 'M', '$\Gamma$']

wg = []
for i in range(len(qq)):
    wg1, null = eigh(Dgeo[i])
    wg.append(wg1)

wg = np.array(wg)
wg = np.transpose(wg)

#plot modes
my_dpi = 96
fig1, ax1 = plt.subplots(1, figsize=(650/my_dpi, 400/my_dpi), dpi=my_dpi)

my_lw = 1.2
my_ms = 1.

for i in range(4):
    plt.plot(qind[1:-1], wg[i,1:-1], 'b-', linewidth=my_lw,markersize = my_ms, zorder=2)

#figure details
ticks = [0, 0.6666, 1., 1.5774]
ticklabels = ['$\Gamma$','K', 'M', '$\Gamma$']
plt.xticks(ticks, ticklabels)
plt.ylabel('Phonon energy  [meV]')
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.9)
#plt.legend(shadow=True, loc = (0.35,0.05), prop={'size': 8})
plt.grid(axis = 'x', linestyle = '--', alpha = 0.6, zorder = -5)

plt.xlim(0, 1.5774)
#plt.ylim(0, 210)
plt.savefig('c:/users/gugli/desktop/tesi/figure/Dgeo_eig'+str(ncont)+'_'+str(nfbz)+'.png', dpi = my_dpi*5)

plt.show()
