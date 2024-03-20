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

#flexural force constants
#alpha_z, gamma_z
alpha_z = -1.176*10**5; gamma_z = 0.190*10**5

#in-plane force constants
#alpha, beta, gamma, delta
alpha = -4.046*10**5; beta = 1.107*10**5;
gamma = -0.238*10**5; delta = -1.096*10**5

#NN vectors (in units of a)
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

#flexural phonon bands
#(wavevector q, force constants A)
#returns 2 eigenfrequencies omega(q)
def Hz(q, alpha_z, gamma_z):
    f1 = 0; f2 = 0
    for i in range(len(d0)):
        f1 = f1 + np.exp(1j*(q[0]*d0[i][0] + q[1]*d0[i][1]))
    for i in range(len(d1)):
        f2 = f2 + np.exp(1j*(q[0]*d1[i][0] + q[1]*d1[i][1]))
    a0 = -3*alpha_z -6*gamma_z
    a1 = f1*alpha_z
    a2 = f2*gamma_z
    #hamiltonian matrix
    h = np.array([[a0+a2, a1],
        [np.conj(a1), a0+a2]])
    w2, null = eigh(h)
    return np.sqrt(w2)

#out-of-plane phonon bands
#(wavevector q, force constants A)
#returns 4 eigenfrequencies omega(q)
def Hxy(q, alpha, beta, gamma, delta):
    a = 2*gamma*(np.cos(np.sqrt(3)*q[1]) + 2*np.cos(3/2*q[0])*np.cos(np.sqrt(3)/2*q[1])-3)-3*alpha
    b = delta*(2*np.cos(np.sqrt(3)*q[1]) +2*np.cos(3/2*q[0]+2*np.pi/3)*np.exp(-1j*np.sqrt(3)/2*q[1]) +2*np.cos(3/2*q[0]-2*np.pi/3)*np.exp(1j*np.sqrt(3)/2*q[1]))
    c = alpha*(np.exp(1j*q[0]) +2*np.exp(-1j/2*q[0])*np.cos(np.sqrt(3)/2*q[1]) )
    d = beta*(np.exp(1j*q[0]) + 2*np.exp(-1j*q[0]/2)*np.cos(np.sqrt(3)/2*q[1]-2*np.pi/3))
    e = beta*(np.exp(1j*q[0]) + 2*np.exp(-1j*q[0]/2)*np.cos(np.sqrt(3)/2*q[1]+2*np.pi/3))
    h = np.array([[a, b, c, d],
                [np.conj(b), a, e, c],
                [np.conj(c), np.conj(e), a, b],
                [np.conj(d), np.conj(c), np.conj(b), a]])
    w2, null = eigh(h)
    return np.sqrt(w2)

#number of points in each part of the contour
points = 1000
#high symmetry contour
kx1 = np.linspace(0, 2*np.pi/3, points)
ky1 = 1/np.sqrt(3)*kx1
k1 = [kx1, ky1]
k1 = np.transpose(k1)
ky2 = np.linspace(2*np.pi/3/np.sqrt(3),0, points)
kx2 = 2*np.pi/3*np.ones(len(ky2))
k2 = [kx2, ky2]
k2 = np.transpose(k2)
kx3 = np.linspace(2*np.pi/3, 0, points)
ky3 = np.zeros(len(kx3))
k3 = [kx3, ky3]
k3 = np.transpose(k3)

#calculate the dispersion on the high-symmetry contour
E1out = []
E1in = []
for i in range(len(k1)):
    e1out = Hz(k1[i], alpha_z, gamma_z)
    e1in = Hxy(k1[i], alpha, beta, gamma, delta)
    E1out.append(e1out)
    E1in.append(e1in)
E2out = []
E2in = []
for i in range(len(k2)):
    e2out = Hz(k2[i], alpha_z, gamma_z)
    e2in = Hxy(k2[i], alpha, beta, gamma, delta)
    E2out.append(e2out)
    E2in.append(e2in)
E3out = []
E3in = []
for i in range(len(k3)):
    e3out = Hz(k3[i], alpha_z, gamma_z)
    e3in = Hxy(k3[i], alpha, beta, gamma, delta)
    E3out.append(e3out)
    E3in.append(e3in)

#convert to numpy array (easier to plot)
E1out = np.array(E1out)
E2out = np.array(E2out)
E3out = np.array(E3out)
E1in = np.array(E1in)
E2in = np.array(E2in)
E3in = np.array(E3in)

#plot modes
my_dpi = 96
fig1, ax1 = plt.subplots(figsize=(500/my_dpi, 400/my_dpi), dpi=my_dpi)

my_lw = 0.7
for i in range(2):
    plt.plot(np.sqrt(kx1**2+ky1**2), E1out[:,i], 'b-', linewidth=my_lw, zorder=2)
    plt.plot(4*np.pi/3/np.sqrt(3)+ np.flip(ky2), E2out[:,i], 'b-', linewidth=my_lw, zorder=2)
    plt.plot(2*np.pi/np.sqrt(3)+ np.flip(kx3), E3out[:,i], 'b-', linewidth=my_lw, zorder = 2)

for i in range(4):
    plt.plot(np.sqrt(kx1**2+ky1**2), E1in[:,i], 'k-', linewidth=my_lw, zorder=2)
    plt.plot(4*np.pi/3/np.sqrt(3)+ np.flip(ky2), E2in[:,i], 'k-', linewidth=my_lw, zorder=2)
    plt.plot(2*np.pi/np.sqrt(3)+ np.flip(kx3), E3in[:,i], 'k-', linewidth=my_lw, zorder = 2)

#figure details
plt.xticks([0, 4*np.pi/3/np.sqrt(3), 6*np.pi/3/np.sqrt(3), 6*np.pi/3/np.sqrt(3) + 2*np.pi/3], ['$\Gamma$','K', 'M', '$\Gamma$'])
plt.ylabel('Frequency [cm$^{-1}$]')
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
#plt.legend(shadow=True, loc = 'upper center', prop={'size': 8})
plt.grid(axis = 'x', linestyle = '--', alpha = 0.5, zorder = -1)
my_fs = 9
plt.text(4.1, 1550, 'TO', fontsize=my_fs)
plt.text(1.2, 1550, 'TO', fontsize=my_fs)
plt.text(4.9, 1370, 'LO', fontsize=my_fs)
plt.text(0.6, 1350, 'LO', fontsize=my_fs)
plt.text(4.5, 1000, 'LA', fontsize=my_fs)
plt.text(1., 1000, 'LA', fontsize=my_fs)
plt.text(1.05, 300, 'TA', fontsize=my_fs)
plt.text(4.3, 300, 'TA', fontsize=my_fs)
plt.text(0.1, 870, 'ZO', fontsize=my_fs)
plt.text(5.3, 870, 'ZO', fontsize=my_fs)
plt.text(1.2, 50, 'ZA', fontsize=my_fs)
plt.text(4.2, 50, 'ZA', fontsize=my_fs)

plt.xlim(0, 5.721994)
plt.ylim(0, 1650)
plt.savefig('c:/users/gugli/desktop/tesi/figure/phonons.jpeg', dpi = my_dpi*5)

plt.show()
