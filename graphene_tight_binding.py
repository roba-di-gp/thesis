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

#hopping integrals (NN, NNN, NNNN)
tt = [-2.97, -0.073, -0.33]
#overlap integrals
ss = [0.073, 0.018, 0.026]

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

#hamiltonian eigenvalues and eigenvectors
#(wavevector k, NN hopping t1, NNN hopping t2)
#returns eigenenergies E(k), eigenvector U(k)
def H(k,e0,t,s):
    #initialize values
    t0 = 0; t1 = 0; t2 = 0; s0 = 0; s1 = 0; s2 = 0
    #NN geometric hopping factor
    for i in range(len(d0)):
        t0 = t0 + t[0]*np.exp(-1j*k[0]*d0[i][0] - 1j*k[1]*d0[i][1])
        s0 = s0 + s[0]*np.exp(-1j*k[0]*d0[i][0] - 1j*k[1]*d0[i][1])
    #NNN geometric hopping factor
    for i in range(len(d1)):
        t1 = t1 + t[1]*np.exp(-1j*k[0]*d1[i][0] - 1j*k[1]*d1[i][1])
        s1 = s1 + s[1]*np.exp(-1j*k[0]*d1[i][0] - 1j*k[1]*d1[i][1])
    #NNNN geometric hopping factor
    for i in range(len(d2)):
        t2 = t2 + t[2]*np.exp(-1j*k[0]*d2[i][0] - 1j*k[1]*d2[i][1])
        s2 = s2 + s[2]*np.exp(-1j*k[0]*d2[i][0] - 1j*k[1]*d2[i][1])
    #hamiltonian matrix
    h = np.array([[e0+t1, t0+t2],
        [np.conj(t0+t2), e0+t1]])
    #overlap matrix
    S = np.array([[1+s1,s0+s2], [np.conj(s0+s2), 1+s1]])
    #diagonalize S
    s, us = eigh(S)
    Sdiag = np.array([[s[0], 0], [0, s[1]]])
    #define S^1/2
    rootSdiag = np.sqrt(Sdiag)
    #transform into the old basis U S^1/2 U*
    rootS = np.matmul(np.matmul((us), rootSdiag), np.transpose(np.conj(us)))
    #calculate equivalent hamiltonian S^-1/2 h S^-1/2
    htilde = np.matmul(np.matmul(inv(rootS), h), inv(rootS))
    #diagonalize equivalent hamiltonian
    e, utilde = eigh(htilde)
    #get the right eigenvectors
    u = np.dot(inv(rootS), utilde)
    return e, u

#draws an hexagon
def draw_hexagon(ax, vertices):
    # Create a Polygon patch with the specified vertices
    hexagon = patches.Polygon(vertices, closed=True, edgecolor='k', facecolor='none',alpha=1, linewidth=0.7)

    # Add the hexagon patch to the Axes
    ax.add_patch(hexagon)

#plot NN, NNN, NNNN vectors
my_dpi = 96
fig, ax = plt.subplots(figsize=(400/my_dpi, 400/my_dpi), dpi=my_dpi)

#draw some hexagons
v1 = np.array([(0,0), (1/2, np.sqrt(3)/2), (3/2, np.sqrt(3)/2),
     (2,0),(3/2, -np.sqrt(3)/2), (1/2, -np.sqrt(3)/2)])
v2 = []; v3 = []; v4 =[]; v5 = []; v6 = []

for i in range(6):
    v2.append(v1[i]+ (0,np.sqrt(3)))
    v3.append(v1[i]+ (-3/2,np.sqrt(3)/2))
    v4.append(v1[i]+ (-3/2,-np.sqrt(3)/2))
    v5.append(v1[i]+ (0,-np.sqrt(3)))
    v6.append(v1[i]+ (-3,0))

draw_hexagon(ax, v1)
draw_hexagon(ax, v2)
draw_hexagon(ax, v3)
draw_hexagon(ax, v4)
draw_hexagon(ax, v5)
draw_hexagon(ax, v6)

#set axis limits
ax.set_xlim(-3.6, 2.8)
ax.set_ylim(-3.2, 3.2)

#NN plot
for i in range(len(d0)):
    plt.plot(d0[i][0], d0[i][1], 'r.',zorder=2, markersize = 7)
    plt.plot([0,d0[i][0]], [0,d0[i][1]], 'r-', linewidth = 1,zorder=2)

plt.plot([0,d0[0][0]], [0,d0[0][1]], 'r-', linewidth = 1,zorder=2, label='NN')

#NNN plot
for i in range(len(d1)):
    plt.plot([0,d1[i][0]], [0,d1[i][1]], 'b.',zorder=3 , markersize = 7)
    plt.plot([0,d1[i][0]], [0,d1[i][1]], 'b--', linewidth = 1, zorder=3)

plt.plot([0,d1[0][0]], [0,d1[0][1]], 'b--', linewidth = 1, zorder=3, label='NNN')

#NNNN plot
for i in range(len(d2)):
    plt.plot([0,d2[i][0]], [0,d2[i][1]], 'r.',zorder=1, markersize = 7)
    plt.plot([0,d2[i][0]], [0,d2[i][1]], 'r:', linewidth = 1, zorder=1)

plt.plot([0,d2[0][0]], [0,d2[0][1]], 'r:', linewidth = 1, zorder=1, label='NNNN')

plt.legend(shadow=True, loc = 'upper left', prop={'size': 8})
plt.xlabel('$x/a$')
plt.ylabel('$y/a$')
plt.subplots_adjust(left=0.15, right=0.85, bottom = 0.15, top = 0.85)

plt.savefig('c:/users/gugli/desktop/tesi/figure/neighbors.jpeg', dpi = my_dpi*5)

#number of points in each part of the contour
points = 300
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

#plot high-symmetry contour
my_dpi = 96
fig0, ax0 = plt.subplots(figsize=(400/my_dpi, 400/my_dpi), dpi=my_dpi)

vertices1 = [
    (0, 4*np.pi/3/np.sqrt(3)),
    (2*np.pi/3, 2*np.pi/3/np.sqrt(3)),
    (2*np.pi/3, -2*np.pi/3/np.sqrt(3)),
    (0, -4*np.pi/3/np.sqrt(3)),
    (-2*np.pi/3, -2*np.pi/3/np.sqrt(3)),
    (-2*np.pi/3, 2*np.pi/3/np.sqrt(3))
]

# Call the function to draw the hexagon
draw_hexagon(ax0, vertices1)
plt.plot(kx1, ky1,'r-')
plt.plot(kx2, ky2,'r-')
plt.plot(kx3, ky3,'r-')
plt.text(-0.45,-0.1,'$\Gamma$')
plt.text(2.25,1.1,'K')
plt.text(2.25,-0.1,'M')

#plt.grid(True, linestyle=':', alpha=0.5)
plt.xlim(-3.7, 3.7)
plt.ylim(-3.7, 3.7)
plt.xlabel('$k_x a$')
plt.ylabel('$k_y a$')
plt.subplots_adjust(left=0.15, right=0.85, bottom = 0.15, top = 0.85)

plt.savefig('c:/users/gugli/desktop/tesi/figure/contour.jpeg', dpi = my_dpi*5)

#plot bands
fig1, ax1 = plt.subplots(figsize=(500/my_dpi, 350/my_dpi), dpi=my_dpi)
my_lw = 0.7

#repeat the calculation for NN, NNN, NNNN and the plot

#NNNN
#calculate the dispersion on the high-symmetry contour
E1 = []
for i in range(len(k1)):
    e1, null = H(k1[i], 0, tt, ss)
    E1.append(e1)
E2 = []
for i in range(len(k2)):
    e2, null = H(k2[i], 0, tt, ss)
    E2.append(e2)
E3 = []
for i in range(len(k3)):
    e3, null = H(k3[i], 0,  tt, ss)
    E3.append(e3)

#convert to numpy array (easier to plot)
E1 = np.array(E1)
E2 = np.array(E2)
E3 = np.array(E3)

for i in range(2):
    plt.plot(np.sqrt(kx1**2+ky1**2), E1[:,i], 'k-', linewidth=my_lw, zorder=4)
    plt.plot(4*np.pi/3/np.sqrt(3)+ np.flip(ky2), E2[:,i], 'k-', linewidth=my_lw, zorder=4)
    plt.plot(2*np.pi/np.sqrt(3)+ np.flip(kx3), E3[:,i], 'k-', linewidth=my_lw, zorder = 4)

plt.plot(2*np.pi/np.sqrt(3)+ np.flip(kx3), E3[:,i], 'k-', linewidth=my_lw, zorder = 4, label = 'NNNN')

#NNN
tt[2] = 0; ss[2] = 0
#calculate the dispersion on the high-symmetry contour
E1 = []
for i in range(len(k1)):
    e1, null = H(k1[i], 0, tt, ss)
    E1.append(e1)
E2 = []
for i in range(len(k2)):
    e2, null = H(k2[i], 0, tt, ss)
    E2.append(e2)
E3 = []
for i in range(len(k3)):
    e3, null = H(k3[i], 0, tt, ss)
    E3.append(e3)

#convert to numpy array (easier to plot)
E1 = np.array(E1)
E2 = np.array(E2)
E3 = np.array(E3)

for i in range(2):
    plt.plot(np.sqrt(kx1**2+ky1**2), E1[:,i], 'b-', linewidth=my_lw, zorder=3)
    plt.plot(4*np.pi/3/np.sqrt(3)+ np.flip(ky2), E2[:,i], 'b-', linewidth=my_lw, zorder=3)
    plt.plot(2*np.pi/np.sqrt(3)+ np.flip(kx3), E3[:,i], 'b-', linewidth=my_lw, zorder = 3)

plt.plot(2*np.pi/np.sqrt(3)+ np.flip(kx3), E3[:,i], 'b-', linewidth=my_lw, zorder = 3, label = 'NNN')

#NN
tt[1] = 0; ss[1] = 0
#calculate the dispersion on the high-symmetry contour
E1 = []
for i in range(len(k1)):
    e1, null = H(k1[i], 0, tt, ss)
    E1.append(e1)
E2 = []
for i in range(len(k2)):
    e2, null = H(k2[i], 0, tt, ss)
    E2.append(e2)
E3 = []
for i in range(len(k3)):
    e3, null = H(k3[i], 0, tt, ss)
    E3.append(e3)

#convert to numpy array (easier to plot)
E1 = np.array(E1)
E2 = np.array(E2)
E3 = np.array(E3)

for i in range(2):
    plt.plot(np.sqrt(kx1**2+ky1**2), E1[:,i], 'r-', linewidth=my_lw, zorder = 2)
    plt.plot(4*np.pi/3/np.sqrt(3)+ np.flip(ky2), E2[:,i], 'r-', linewidth=my_lw, zorder=2)
    plt.plot(2*np.pi/np.sqrt(3)+ np.flip(kx3), E3[:,i], 'r-', linewidth=my_lw, zorder = 2)

plt.plot(2*np.pi/np.sqrt(3)+ np.flip(kx3), E3[:,i], 'r-', linewidth=my_lw, zorder = 2, label = 'NN')

#figure details
plt.xticks([0, 4*np.pi/3/np.sqrt(3), 6*np.pi/3/np.sqrt(3), 6*np.pi/3/np.sqrt(3) + 2*np.pi/3], ['$\Gamma$','K', 'M', '$\Gamma$'])
plt.ylabel('Energy [eV]')
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.15, top = 0.85)
plt.legend(shadow=True, loc = 'upper center', prop={'size': 8})
plt.grid(axis = 'x', linestyle = '--', alpha = 0.5, zorder = -1)
plt.xlim(0, 5.721994)

plt.savefig('c:/users/gugli/desktop/tesi/figure/pi_bands.jpeg', dpi = my_dpi*5)


plt.show()