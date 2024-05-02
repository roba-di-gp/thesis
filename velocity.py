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
from functions import *

#hbar [eV s]
hbar = 6.582119569e-16
#lattice parameter [angstrom]
lp = 2.466731
#lattice step [angstrom]
a = lp/np.sqrt(3)

#best fit parameters [eV]
pp = np.load('c:/users/gugli/desktop/tesi/data/bande_bestfit.npy')

#energy of the isolated 2p_z electron
epz = pp[0]
#hopping integrals (NN, NNN, NNNN)
tt = pp[1:4]
#overlap integrals
ss = pp[4:7]

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

#number of points in the contour
points = 1000
length = 1+0.5+np.cos(np.pi/6)
len1 = int(1/length*points)
len2 = int(0.5/length*points)
len3 = int(np.cos(np.pi/6)/length*points)
#high symmetry contour
kx1 = np.linspace(0, 2*np.pi/3, len1)
ky1 = 1/np.sqrt(3)*kx1
k1 = [kx1, ky1]
k1 = np.transpose(k1)
ky2 = np.linspace(2*np.pi/3/np.sqrt(3),0, len2)
kx2 = 2*np.pi/3*np.ones(len(ky2))
k2 = [kx2, ky2]
k2 = np.transpose(k2)
kx3 = np.linspace(2*np.pi/3, 0, len3)
ky3 = np.zeros(len(kx3))
k3 = [kx3, ky3]
k3 = np.transpose(k3)

k1dir = []
for i in range(len(k1)):
    if i == 0:
        k1dir.append(k1[i+1]-k1[i])
    if i == len(k1)-1:
        break
    else:
        k1dir.append((k1[i+1]-k1[i]))
k2dir = []
for i in range(len(k2)):
    if i == 0:
        k2dir.append(k2[i+1]-k2[i])
    if i == len(k2)-1:
        break
    else:
        k2dir.append((k2[i+1]-k2[i]))
k3dir = []
for i in range(len(k3)):
    if i == 0:
        k3dir.append(k3[i+1]-k3[i])
    if i == len(k3)-1:
        break
    else:
        k3dir.append((k3[i+1]-k3[i]))

#plot bands
my_dpi = 96
fig1, ax1 = plt.subplots(figsize=(300*(1+np.sqrt(5))/2/my_dpi, 300/my_dpi), dpi=my_dpi)
my_lw = 1.1

def vg(k, e0, t, s):
    e = np.sqrt(3+2*np.cos(np.sqrt(3)*k[1])+ 4*np.cos(np.sqrt(3)/2*k[1])*np.cos(3/2*k[0]))
    vgx = -3/e*np.cos(np.sqrt(3)/2*k[1])*np.sin(3/2*k[0])
    vgy = -np.sqrt(3)/e*(np.sin(np.sqrt(3)*k[1]) + np.sin(np.sqrt(3)/2*k[1])*np.cos(3/2*k[0]))
    vg = a*tt[0]*np.array([np.array([vgx, -vgx]), np.array([vgy, -vgy])])
    return vg

#NNNN
#calculate the dispersion on the high-symmetry contour
#calculate the gradient
dk = [1e-3, 1e-3]
E1 = []
v1 = []
for i in range(len(k1)):
    e1, u1, p1 = EUP(k1[i], epz, tt, ss)
    vv1, dp1 = dEdP(k1[i], epz, tt, ss, dk)
    E1.append(e1)
    v1.append(vv1)
E2 = []
v2 = []
for i in range(len(k2)):
    e2, u2, p2 = EUP(k2[i], epz, tt, ss)
    vv2, dp2 = dEdP(k2[i], epz, tt, ss, dk)
    E2.append(e2)
    v2.append(vv2)
E3 = []
v3 = []
for i in range(len(k3)):
    e3, u3, p3 = EUP(k3[i], epz,  tt, ss)
    vv3, dp3 = dEdP(k3[i], epz, tt, ss, dk)
    E3.append(e3)
    v3.append(vv3)

#convert to numpy array (easier to plot)
E1 = np.array(E1)
E2 = np.array(E2)
E3 = np.array(E3)
v1 = np.array(v1)/hbar*1e-16
v2 = np.array(v2)/hbar*1e-16
v3 = np.array(v3)/hbar*1e-16

k1dir = np.array(k1dir)
k2dir = np.array(k2dir)
k3dir = np.array(k3dir)

mark = ['k-', 'r-']
for i in range(2):
    ax1.plot(np.sqrt(kx1**2+ky1**2), E1[:,i], mark[i], linewidth=my_lw, zorder=4)
    ax1.plot(4*np.pi/3/np.sqrt(3)+ np.flip(ky2), E2[:,i], mark[i], linewidth=my_lw, zorder=4)
    ax1.plot(2*np.pi/np.sqrt(3)+ np.flip(kx3), E3[:,i], mark[i],linewidth=my_lw, zorder = 4)

#plot the directional derivative
k1dmod = np.sqrt(k1dir[:,0]**2+k1dir[:,1]**2)
k2dmod = np.sqrt(k2dir[:,0]**2+k2dir[:,1]**2)
k3dmod = np.sqrt(k3dir[:,0]**2+k3dir[:,1]**2)
ax2 = ax1.twinx()

mark = ['k--', 'r--']
for i in range(2):
    ax2.plot(np.sqrt(kx1**2+ky1**2), (v1[:,0,i]*k1dir[:,0] + v1[:,1,i]*k1dir[:,1])/k1dmod, mark[i], linewidth=my_lw, zorder=4)
    ax2.plot(4*np.pi/3/np.sqrt(3)+ np.flip(ky2), (v2[:,0,i]*k2dir[:,0] + v2[:,1,i]*k2dir[:,1])/k2dmod, mark[i], linewidth=my_lw, zorder=4)
    ax2.plot(2*np.pi/np.sqrt(3)+ np.flip(kx3), (v3[:,0,i]*k3dir[:,0] + v3[:,1,i]*k3dir[:,1])/k3dmod, mark[i], linewidth=my_lw, zorder = 4)

#figure details
ax1.set_xticks([0, 4*np.pi/3/np.sqrt(3), 6*np.pi/3/np.sqrt(3), 6*np.pi/3/np.sqrt(3) + 2*np.pi/3])
ax1.set_xticklabels(['$\Gamma$','K', 'M', '$\Gamma$'])
ax1.set_ylabel('Energy  [eV]')
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.15, top = 0.85)
#ax1.legend(shadow=True, loc = 'upper center', prop={'size': 8})
ax1.grid(axis = 'x', linestyle = '--', alpha = 0.6, zorder = -2)
ax1.set_xlim(0, 5.721994)
ax1.set_ylim(-12,12)
ax2.set_ylabel('Group velocity [$10^6$ m/s]',rotation = -90, labelpad = 15)

plt.savefig('c:/users/gugli/desktop/tesi/figure/velocity.jpeg', dpi = my_dpi*5)

plt.show()
