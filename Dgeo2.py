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
from time import time

#numbers
#c [nm/fs]
c = 299.792458
#hbar [eV fs]
hbar = 0.6582119569
#lattice parameter (from Guido) [nm]
lp = 0.2466731
#lattice step [nm]
a = lp/np.sqrt(3)
#lattice parameter/lattice step conversion factor * 2pi
conv_lattice = 2*np.pi/np.sqrt(3)
#carbon atom mass [eV fs^2 / nm^2]
M = 12*0.93149e9/c**2

print(' Packages and functions imported \n')

#best fit parameters for electron bands [eV]
pp_el = np.load('c:/users/gugli/desktop/tesi/data/bande_bestfit.npy')
#best fit parameters for phonon lines [eV^2]
pp_ph = np.load(r'c:/users/gugli/desktop/tesi/data/phonons_bestfit.npy')
#gamma parameters [nm^-2]
gamma = np.load(r'c:/users/gugli/desktop/tesi/data/gamma.npy')
gamma_mat = np.array([np.array([gamma[0], gamma[1]]), np.array([gamma[1], gamma[0]])])
#from Angstrom^-2 to nm^-2
gamma_mat = gamma_mat*100

print(' Data imported \n')

#build the high-symmetry contour
#number of points in the contour (times x 91)
times = 3
lenGK = 41*times
lenKM = 21*times
lenMG = 31*times
#high symmetry contour
qxMG = np.linspace(2*np.pi/3, 0, lenMG)
qyMG = np.zeros(lenMG)
qxGK = np.linspace(0, 2*np.pi/3, lenGK)
qyGK = 1/np.sqrt(3)*qxGK
qyKM = np.linspace(2*np.pi/3/np.sqrt(3),0, lenKM)
qxKM = 2*np.pi/3*np.ones(lenKM)

qindGK = np.sqrt(qxGK**2+qyGK**2)/conv_lattice
qindKM = qindGK[-1] + np.flip(qyKM)/conv_lattice
qindMG = qindKM[-1] + np.flip(qxMG)/conv_lattice

qqx = np.concatenate((qxGK, qxKM[1:], qxMG[1:]))
qqy = np.concatenate((qyGK, qyKM[1:], qyMG[1:]))

qind = np.concatenate((qindGK, qindKM[1:], qindMG[1:]))

for i in range(len(qind)):
    qind[i] = round(qind[i], 4)

qq = np.array([qqx, qqy])
qq = np.transpose(qq)

np.save('c:/users/gugli/desktop/tesi/data/ph_contour_vecs_277.npy', qq)
np.save('c:/users/gugli/desktop/tesi/data/ph_contour_ind_277.npy', qind)
print(' Contour length = %.0f \n'%(len(qq)))

##evaluation for a single wavevector

q = [0., 0.]

#FBZ meshgrid
fbzmesh = fun.fbz_meshgrid(12)
#differential
dkx = 2*np.pi/3/np.sqrt(len(fbzmesh))/1000
dky = 2*np.pi/np.sqrt(3)/np.sqrt(len(fbzmesh))/1000
my_dk = np.array([dkx, dky])
dgeo = fun.calc_DGeo(q, fbzmesh, pp_el, gamma_mat, my_dk, 0.001)*1e6 #[meV^2]
print(' Re[D(q)]: ')
print(np.round(np.real(dgeo), 5))
print('\n')
print(' Im[D(q)]: ')
print(np.round(np.imag(dgeo), 3))

##evaluation on the complete contour

#FBZ meshgrid
fbzmesh = fun.fbz_meshgrid(48)

my_gap = 0.001 #[eV]

print(' FBZ grid length = %.0f x %.0f \n'%(np.sqrt(len(fbzmesh)), np.sqrt(len(fbzmesh))))
print(' Save data? \n')
save = input()
save_data = False
if save == 'y':
    save_data = True
if save_data:
    print('\n Saving data \n')

start = time()

#differential
dkx = 2*np.pi/3/np.sqrt(len(fbzmesh))/100
dky = 2*np.pi/np.sqrt(3)/np.sqrt(len(fbzmesh))/100
my_dk = np.array([dkx, dky])

DGeo = []
i = 0
for q in qq:
    timezero = time()
    dgeo = fun.calc_DGeo(q, fbzmesh, pp_el, gamma_mat, my_dk, my_gap)
    DGeo.append(dgeo)
    if i == 0:
        print('\n Estimated time: %.1f min \n '%((time()-timezero)*len(qq)/60))
    else:
        print(' %.2f'%(i/len(qq)))
    i+=1

if save_data:
    np.save('c:/users/gugli/desktop/tesi/data/Dgeo_277_30_1mev.npy', DGeo)

print('\n Elapsed time: %.1f min\n'%((time()-start)/60))

plt.figure()
plt.text(0.05, 0.1,'Finito!', size = 50)
plt.xlim(0,0.3)
plt.ylim(0,0.3)
plt.show()

##Gamma point neighbors (right)

lenGK = 50
ext = 0.01
qxGK = np.linspace(0, ext, lenGK)
qyGK = 1/np.sqrt(3)*qxGK
qindGK = np.sqrt(qxGK**2+qyGK**2)/conv_lattice
qqGK = np.array([qxGK, qyGK])
qqGK = np.transpose(qqGK)

#FBZ meshgrid
fbzmesh = fun.fbz_meshgrid(30)

print(' FBZ grid length = %.0f x %.0f\n'%(np.sqrt(len(fbzmesh)), np.sqrt(len(fbzmesh))))
print(' Save data? \n')
save = input()
save_data = False
if save == 'y':
    save_data = True
if save_data:
    print('\n Saving data \n')

start = time()

#differential
dkx = 2*np.pi/3/np.sqrt(len(fbzmesh))/1000
dky = 2*np.pi/np.sqrt(3)/np.sqrt(len(fbzmesh))/1000
my_dk = np.array([dkx, dky])

DGeo = []
i = 0
my_gap = 0.001
for q in qqGK:
    timezero = time()
    dgeo = fun.calc_DGeo(q, fbzmesh, pp_el, gamma_mat, my_dk, my_gap)
    DGeo.append(dgeo)
    if i == 0:
        print('\n Estimated time: %.1f min \n '%((time()-timezero)*len(qqGK)/60))
    else:
        print(' %.2f'%(i/lenGK))
    i+=1

if save_data:
    np.save('c:/users/gugli/desktop/tesi/data/qGK.npy', qqGK)
    np.save('c:/users/gugli/desktop/tesi/data/qGKind.npy', qindGK)
    np.save('c:/users/gugli/desktop/tesi/data/Dgeo_GK.npy', DGeo)

print('\n Elapsed time: %.1f min\n'%((time()-start)/60))

plt.figure()
plt.text(0.05, 0.1,'Finito', size = 50)
plt.xlim(0,0.3)
plt.ylim(0,0.3)
plt.show()

##Gamma point neighbors (left)

lenMG = 70
ext = 0.02
qxMG = np.linspace(ext, 0, lenMG)
qyMG = np.zeros(lenMG)
qindMG = 1.5774 - qxMG/conv_lattice
qqMG = np.array([qxMG, qyMG])
qqMG = np.transpose(qqMG)

#FBZ meshgrid
fbzmesh = fun.fbz_meshgrid(30)

print(' FBZ grid length = %.0f x %.0f\n'%(len(fbzmesh), len(fbzmesh)))
print(' Save data? \n')
save = input()
save_data = False
if save == 'y':
    save_data = True
if save_data:
    print('\n Saving data \n')

start = time()

#differential
dkx = 2*np.pi/3/np.sqrt(len(fbzmesh))/1000
dky = 2*np.pi/np.sqrt(3)/np.sqrt(len(fbzmesh))/1000
my_dk = np.array([dkx, dky])

my_gap = 0.001
DGeo = []
i = 0
for q in qqMG:
    timezero = time()
    dgeo = fun.calc_DGeo(q, fbzmesh, pp_el, gamma_mat, my_dk, my_gap)
    DGeo.append(dgeo)
    if i == 0:
        print('\n Estimated time: %.1f min \n '%((time()-timezero)*len(qqMG)/60))
    else:
        print(' %.2f'%(i/lenMG))
    i+=1

if save_data:
    np.save('c:/users/gugli/desktop/tesi/data/qMG.npy', qqMG)
    np.save('c:/users/gugli/desktop/tesi/data/qMGind.npy', qindMG)
    np.save('c:/users/gugli/desktop/tesi/data/Dgeo_MG.npy', DGeo)

print('\n Elapsed time: %.1f min\n'%((time()-start)/60))

plt.figure()
plt.text(0.05, 0.1,'Finito', size = 50)
plt.xlim(0,0.3)
plt.ylim(0,0.3)
plt.show()

##M point neighbors

#build the high-symmetry contour
#number of points in the contour (times x 91)
lenM = 35 #(x2)
ext = 0.02
#high symmetry contour
qyKM = np.linspace(ext, 0, lenM)
qxKM = 2*np.pi/3*np.ones(lenM)
qxMG = np.linspace(2*np.pi/3, 2*np.pi/3-ext, lenM+1)
qyMG = np.zeros(lenM+1)

qindKM = 1 - qyKM/conv_lattice
qindMG = qindKM[-1] + (2*np.pi/3 - qxMG)/conv_lattice

qqxM = np.concatenate((qxKM, qxMG[1:]))
qqyM = np.concatenate((qyKM, qyMG[1:]))

qindM = np.concatenate((qindKM, qindMG[1:]))

qqM = np.array([qqxM, qqyM])
qqM = np.transpose(qqM)

#FBZ meshgrid
fbzmesh = fun.fbz_meshgrid(42)

print(' FBZ grid length = %.0f x %.0f\n'%(len(fbzmesh), len(fbzmesh)))
print(' Save data? \n')
save = input()
save_data = False
if save == 'y':
    save_data = True
if save_data:
    print('\n Saving data \n')

start = time()

#differential
dkx = 2*np.pi/3/np.sqrt(len(fbzmesh))/1000
dky = 2*np.pi/np.sqrt(3)/np.sqrt(len(fbzmesh))/1000
my_dk = np.array([dkx, dky])

DGeo = []
i = 0
for q in qqM:
    timezero = time()
    dgeo = fun.calc_DGeo(q, fbzmesh, pp_el, gamma_mat, my_dk)
    DGeo.append(dgeo)
    if i == 0:
        print('\n Estimated time: %.1f min \n '%((time()-timezero)*len(qqM)/60))
    else:
        print(' %.2f'%(i/lenM/2))
    i+=1

if save_data:
    np.save('c:/users/gugli/desktop/tesi/data/qM.npy', qqM)
    np.save('c:/users/gugli/desktop/tesi/data/qMind.npy', qindM)
    np.save('c:/users/gugli/desktop/tesi/data/Dgeo_M.npy', DGeo)

print('\n Elapsed time: %.1f min\n'%((time()-start)/60))

plt.figure()
plt.text(0.05, 0.1,'Finito', size = 50)
plt.xlim(0,0.3)
plt.ylim(0,0.3)
plt.show()

##K point neighbors

#build the high-symmetry contour
lenK = 35 #(x2)
ext = 0.02
qxGK = np.linspace(2*np.pi/3-ext, 2*np.pi/3, lenK)
qyGK = 1/np.sqrt(3)*qxGK
qyKM = np.linspace(2*np.pi/3/np.sqrt(3),2*np.pi/3/np.sqrt(3)-ext, lenK+1)
qxKM = 2*np.pi/3*np.ones(lenK+1)

qindGK = np.sqrt(qxGK**2+qyGK**2)/conv_lattice
qindKM = qindGK[-1] + (2*np.pi/3/np.sqrt(3) - qyKM)/conv_lattice

qqxK = np.concatenate((qxGK, qxKM[1:]))
qqyK = np.concatenate((qyGK, qyKM[1:]))

qindK = np.concatenate((qindGK, qindKM[1:]))

qqK = np.array([qqxK, qqyK])
qqK = np.transpose(qqK)

#FBZ meshgrid
fbzmesh = fun.fbz_meshgrid(42)

print(' FBZ grid length = %.0f x %.0f\n'%(len(fbzmesh), len(fbzmesh)))
print(' Save data? \n')
save = input()
save_data = False
if save == 'y':
    save_data = True
if save_data:
    print('\n Saving data \n')

start = time()

#differential
dkx = 2*np.pi/3/np.sqrt(len(fbzmesh))/1000
dky = 2*np.pi/np.sqrt(3)/np.sqrt(len(fbzmesh))/1000
my_dk = np.array([dkx, dky])

DGeo = []
i = 0
for q in qqK:
    timezero = time()
    dgeo = fun.calc_DGeo(q, fbzmesh, pp_el, gamma_mat, my_dk)
    DGeo.append(dgeo)
    if i == 0:
        print('\n Estimated time: %.1f min \n '%((time()-timezero)*len(qqK)/60))
    else:
        print(' %.2f'%(i/lenK/2))
    i+=1

if save_data:
    np.save('c:/users/gugli/desktop/tesi/data/qK.npy', qqK)
    np.save('c:/users/gugli/desktop/tesi/data/qKind.npy', qindK)
    np.save('c:/users/gugli/desktop/tesi/data/Dgeo_K.npy', DGeo)

print('\n Elapsed time: %.1f min\n'%((time()-start)/60))

plt.figure()
plt.text(0.05, 0.1,'Finito', size = 50)
plt.xlim(0,0.3)
plt.ylim(0,0.3)
plt.show()


