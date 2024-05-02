import numpy
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

#energy of the isolated 2p_z electron
epz = -0.28
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

lw = 0.7
draw_hexagon(ax, v1, lw)
draw_hexagon(ax, v2, lw)
draw_hexagon(ax, v3, lw)
draw_hexagon(ax, v4, lw)
draw_hexagon(ax, v5, lw)
draw_hexagon(ax, v6, lw)

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
plt.show()
