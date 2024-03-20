import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
matplotlib.rcParams.update({'font.size': 15})
from time import time
start = time()

#build a NxN primitive cell (PC) grid
#final grid has approx NxN wavevectors
#wavevectors are in units of 1/a

#set N
N = 12

#draw FBZ contour for reference

vertices1 = [
    (0, 4*np.pi/3/np.sqrt(3)),
    (2*np.pi/3, 2*np.pi/3/np.sqrt(3)),
    (2*np.pi/3, -2*np.pi/3/np.sqrt(3)),
    (0, -4*np.pi/3/np.sqrt(3)),
    (-2*np.pi/3, -2*np.pi/3/np.sqrt(3)),
    (-2*np.pi/3, 2*np.pi/3/np.sqrt(3))
]

def draw_hexagon(ax, vertices):
    # Create a Polygon patch with the specified vertices
    hexagon = patches.Polygon(vertices, closed=True, edgecolor='black', facecolor='none', linewidth=0.6)

    # Add the hexagon patch to the Axes
    ax.add_patch(hexagon)

# Create a figure and axis
my_dpi = 96
fig, ax = plt.subplots(figsize=(400/my_dpi, 400/my_dpi), dpi=my_dpi)

# Call the function to draw the hexagon
draw_hexagon(ax, vertices1)
plt.grid(True, linestyle=':', alpha=0.5)

#primitive vectors
b1 = np.array([2*np.pi/3, 2*np.pi/np.sqrt(3)])
b2 = np.array([2*np.pi/3, -2*np.pi/np.sqrt(3)])

#unit cell contour
plt.plot([0,b1[0]],[0,b1[1]],'r-', linewidth=1)
plt.plot([0,b2[0]],[0,b2[1]],'r-', linewidth=1)
plt.plot([b1[0],2*b1[0]],[b1[1],0],'r-', linewidth=1)
plt.plot([b2[0],2*b2[0]],[b2[1],0],'r-', linewidth=1)

#non-primitive reciprocal lattice vectors (we only need these three)
bb = np.zeros((4,2), dtype='float')
#changing the order of these changes the symmetry structure of the FBZ meshgrid
bb[0] = np.zeros(2); bb[1] = -b1; bb[2] = -b1-b2; bb[3] = -b2

#projectors on reciprocal vectors
p = np.zeros((4,2,2), dtype='float')
for i in range(4):
    if bb[i][0] == 0:
        p[i] = np.array([np.array([0,0]), np.array([0,0])])
    else:
        p[i,:,:] = np.outer(np.transpose(bb[i])/np.hypot(bb[i][0],bb[i][1]),bb[i]/np.hypot(bb[i][0],bb[i][1]))

#build the grid on the PC
kpc = []
for i in range(N):
    for j in range(N):
        kpc.append(i/N*b1 + j/N*b2)

#plot grid on the PC
for l in range(N*N):
    plt.plot(kpc[l][0], kpc[l][1],'+', markersize = 4, color = 'blue')

#fold on the FBZ
#if eps = 0, we get no points on the contour, symmetric FBZ meshgrid with sum(k) = 0; otherwise, if eps = fraction of the spacing, we get an asymmetric FBZ meshgrid with sum(k) \neq 0
eps = np.hypot(b1[0], b1[1])/N/10
k = []
for l in range(N*N):
    for m in range(4):
        pr = []
        kk = kpc[l]+bb[m]
        for n in range(4):
            pr.append(np.hypot(p[n,0,0]*kk[0] + p[n,0,1]*kk[1],
            p[n,1,0]*kk[0] + p[n,1,1]*kk[1]))
        if pr[0] < (b1[0] + eps) and pr[1] < (b1[0] + eps) and pr[2] < (b1[0] + eps) and pr[3] < (b1[0] + eps):
            k.append(kk)
            break

#convert to numpy array and save as a .npy file
k = np.array(k)
np.save('c:/users/gugli/desktop/tesi/codice/fbz_meshgrid.npy',k)

#calc time
print('%.2f'%(time()-start))

#plot the meshgrid

plt.plot(k[:,0], k[:,1],'kx', markersize = 4)

plt.xlim(-4.5, 4.5)
plt.ylim(-4.5, 4.5)
plt.xlabel('$k_x a$')
plt.ylabel('$k_y a$')
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)

plt.savefig(r'c:\users\gugli\desktop\tesi\figure\meshgrid.jpeg', dpi = my_dpi*5)


plt.show()