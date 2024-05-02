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

#draws an hexagon
def draw_hexagon(ax, vertices, lw):
    # Create a Polygon patch with the specified vertices
    hexagon = patches.Polygon(vertices, closed=True, edgecolor='k', facecolor='none',alpha=1, linewidth=lw)

    # Add the hexagon patch to the Axes
    ax.add_patch(hexagon)

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

lw = 1.3
lw1 = 1.5
draw_hexagon(ax0, vertices1, lw)
plt.plot(kx1, ky1,'r-', linewidth= lw1)
plt.arrow(kx1[0],ky1[0], 1, 1/np.sqrt(3), head_width=0.18, facecolor= 'r', edgecolor = 'r')
plt.plot(kx2, ky2,'r-', linewidth= lw1)
plt.arrow(kx2[0],ky2[0], 0., -0.5, head_width=0.18, facecolor= 'r', edgecolor = 'r')
plt.plot(kx3, ky3,'r-', linewidth= lw1)
plt.arrow(kx3[0],ky3[0], -0.8, 0., head_width=0.18, facecolor= 'r', edgecolor = 'r')
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
plt.show()