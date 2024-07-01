import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
matplotlib.rcParams.update({'font.size': 15})
import os
os.chdir('c:/users/gugli/desktop/tesi/codice')
import functions as fun
from time import time
start = time()

#primitive vectors
b1 = np.array([2*np.pi/3, 2*np.pi/np.sqrt(3)])
b2 = np.array([2*np.pi/3, -2*np.pi/np.sqrt(3)])
S = 1
q0 = [2*np.pi/3, -2*np.pi/3/np.sqrt(3)]
k = fun.fbz_meshgrid(18, S, 0)
my_ms = 10.7*S
print('sum(k) = (%.E, %.E) [1/a]'%(sum(k)[0], sum(k)[1]))

print('len(k) = %.0f'%(len(k)))

#calc time
print('calc time = %.2f s'%(time()-start))


# Create a figure and axis
my_dpi = 96
fig, ax = plt.subplots(figsize=(450/my_dpi, 450/my_dpi), dpi=my_dpi)
ax.set_aspect('equal', 'box')
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
    hexagon = patches.Polygon(vertices, closed=True, edgecolor='black', facecolor='none', linewidth=0.6, zorder=-2, alpha = 0.7)

    # Add the hexagon patch to the Axes
    ax.add_patch(hexagon)

# Call the function to draw the hexagon
draw_hexagon(ax, vertices1)
#plt.grid(True, linestyle=':', alpha=0.5)

'''
#plot primitive cell meshgrid with numbers
plt.plot(kpc1[:,0], kpc1[:,1], marker = '$1$', color = 'k', markersize=5, zorder=-1, linewidth=0)
plt.plot(kpc2[:,0], kpc2[:,1], marker = '$2$', color = 'k', markersize=5, zorder=-1, linewidth=0)
plt.plot(kpc3[:,0], kpc3[:,1], marker = '$3$', color = 'k', markersize=5, zorder=-1, linewidth=0)
plt.plot(kpc4[:,0], kpc4[:,1], marker = '$4$', color = 'k', markersize=5, zorder=-1, linewidth=0)

#plot FBZ meshgrid with numbers
plt.plot(k1[:,0], k1[:,1], marker = '$1$', color = 'k', markersize=5, zorder=-1, linewidth=0)
plt.plot(k2[:,0], k2[:,1], marker = '$2$', color = 'k', markersize=5, zorder=-1, linewidth=0)
plt.plot(k3[:,0], k3[:,1], marker = '$3$', color = 'k', markersize=5, zorder=-1, linewidth=0)
plt.plot(k4[:,0], k4[:,1], marker = '$4$', color = 'k', markersize=5, zorder=-1, linewidth=0)
'''

#plot primitive cell meshgrid
#plt.plot(kpc[:,0], kpc[:,1], marker = '$x$', color = 'k', markersize=5, zorder=-1, linewidth=0)

#plot FBZ meshgrid
plt.plot(k[:,0], k[:,1], marker = 'h', color = 'k', markersize=my_ms, zorder=-1, linewidth=0, alpha=1)
'''
plt.plot(b1[0], -b1[1]/3, marker = 'h', color = 'r', markersize=my_ms, zorder=3, linewidth=0, alpha=1)
plt.plot(-b1[0], b1[1]/3, marker = 'h', color = 'r', markersize=my_ms, zorder=3, linewidth=0, alpha=1)
plt.plot(-b1[0], 0, marker = 'h', color = 'r', markersize=my_ms, zorder=3, linewidth=0, alpha=1)
plt.plot(b1[0]/2, b1[1]/2, marker = 'h', color = 'r', markersize=my_ms, zorder=3, linewidth=0, alpha=1)
plt.plot(b1[0]/2, -b1[1]/2, marker = 'h', color = 'r', markersize=my_ms, zorder=3, linewidth=0, alpha=1)
plt.plot(0, 0, marker = 'h', color = 'r', markersize=my_ms, zorder=3, linewidth=0, alpha=1)
'''
# #unit cell contour
# plt.plot([0,b1[0]],[0,b1[1]],'k-', linewidth=0.8, zorder=-2, alpha=0.5)
# plt.plot([0,b2[0]],[0,b2[1]],'k-', linewidth=0.8, zorder=-2, alpha=0.5)
# plt.plot([b1[0],2*b1[0]],[b1[1],0],'k-', linewidth=0.8, zorder=-2, alpha=0.5)
# plt.plot([b2[0],2*b2[0]],[b2[1],0],'k-', linewidth=0.8, zorder=-2, alpha=0.5)

lim = 2.6
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.xlabel('$k_x a$')
plt.ylabel('$k_y a$')
plt.subplots_adjust(left=0.15, right=0.9, bottom = 0, top = 1)
ax.set_aspect('equal', 'box')
plt.savefig(r'c:\users\gugli\desktop\tesi\figure\meshgrid.jpeg', dpi = my_dpi*5)
plt.show()