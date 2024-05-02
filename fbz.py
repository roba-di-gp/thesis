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

#build a NxN meshgrid on the FBZ
#wavevectors are in units of 1/a
N = 12 #set N
plot = 0 #if True, plots meshgrid

#primitive vectors
b1 = np.array([2*np.pi/3, 2*np.pi/np.sqrt(3)])
b2 = np.array([2*np.pi/3, -2*np.pi/np.sqrt(3)])

#build the grid on the PC
kpc = []
for i in range(N):
    for j in range(N):
        kpc.append(i/N*b1 + j/N*b2)

#fold on the FBZ
    # Create a figure and axis
my_dpi = 96
fig, ax = plt.subplots(figsize=(500/my_dpi, 500/my_dpi), dpi=my_dpi)

kpc1 = []; kpc2 = []; kpc3 = []; kpc4 = []
k1 = []; k2 = []; k3 = []; k4 = []
eps = np.hypot(b1[0],b1[1])/N/10
for l in range(N*N):
    #bulk 1
    if kpc[l][0] < 2*np.pi/3-eps and kpc[l][1] < 4*np.pi/3/np.sqrt(3) - 1/np.sqrt(3)*kpc[l][0] +eps and kpc[l][1] > -4*np.pi/3/np.sqrt(3) + 1/np.sqrt(3)*kpc[l][0] +eps:
        kk = kpc[l]
        kpc1.append(kpc[l])
        k1.append(kk)
    #right border 1
    if kpc[l][0] > 2*np.pi/3-eps and kpc[l][0] <  2*np.pi/3 +eps and kpc[l][1] < 4*np.pi/3/np.sqrt(3) - 1/np.sqrt(3)*kpc[l][0] +eps and kpc[l][1] > -4*np.pi/3/np.sqrt(3) + 1/np.sqrt(3)*kpc[l][0] +eps:
        kk = kpc[l]-b1-b2
        kpc1.append(kpc[l])
        k1.append(kk)
    #lower border 1
    if kpc[l][0] <  2*np.pi/3 +eps and kpc[l][1] > -4*np.pi/3/np.sqrt(3) + 1/np.sqrt(3)*kpc[l][0] -eps and kpc[l][1] < -4*np.pi/3/np.sqrt(3) + 1/np.sqrt(3)*kpc[l][0] +eps:
        kk = kpc[l]
        kpc1.append(kpc[l])
        k1.append(kk)
    #2
    if kpc[l][1] > 4*np.pi/3/np.sqrt(3) - 1/np.sqrt(3)*kpc[l][0] + eps and kpc[l][1] >  1/np.sqrt(3)*kpc[l][0]-eps:
        kk = kpc[l]-b1
        kpc2.append(kpc[l])
        k2.append(kk)
    #3
    if kpc[l][1] < -4*np.pi/3/np.sqrt(3) + 1/np.sqrt(3)*kpc[l][0]-eps and kpc[l][1] <  -1/np.sqrt(3)*kpc[l][0]+eps:
        kk = kpc[l]-b2
        kpc3.append(kpc[l])
        k3.append(kk)
    #4
    if kpc[l][0] > 2*np.pi/3 + eps and kpc[l][1] < 1/np.sqrt(3)*kpc[l][0] - eps and kpc[l][1] > -1/np.sqrt(3)*kpc[l][0]+eps:
        kk = kpc[l]-b1-b2
        kpc4.append(kpc[l])
        k4.append(kk)


kpc1 = np.array(kpc1); kpc2 = np.array(kpc2);
kpc3 = np.array(kpc3); kpc4 = np.array(kpc4)

k1 = np.array(k1); k2 = np.array(k2);
k3 = np.array(k3); k4 = np.array(k4)

kpc = np.concatenate((kpc1, kpc2), axis = 0)
kpc = np.concatenate((kpc, kpc3), axis = 0)
kpc = np.concatenate((kpc, kpc4), axis = 0)

k = np.concatenate((k1, k2), axis = 0)
k = np.concatenate((k, k3), axis = 0)
k = np.concatenate((k, k4), axis = 0)

np.save('c:/users/gugli/desktop/tesi/data/fbz_meshgrid.npy',k)

print('sum(k) = (%.E, %.E) [1/a]'%(sum(k)[0], sum(k)[1]))

print('len(k) = %.0f'%(len(k)))

#calc time
print('calc time = %.2f s'%(time()-start))

#plot the meshgrid
if plot:

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
        hexagon = patches.Polygon(vertices, closed=True, edgecolor='black', facecolor='none', linewidth=0.6, zorder=-2)

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
    plt.plot(k[:,0], k[:,1], marker = 'x', color = 'k', markersize=4.5, zorder=-1, linewidth=0)

    #unit cell contour
    plt.plot([0,b1[0]],[0,b1[1]],'r-', linewidth=0.8, zorder=-2)
    plt.plot([0,b2[0]],[0,b2[1]],'r-', linewidth=0.8, zorder=-2)
    plt.plot([b1[0],2*b1[0]],[b1[1],0],'r-', linewidth=0.8, zorder=-2)
    plt.plot([b2[0],2*b2[0]],[b2[1],0],'r-', linewidth=0.8, zorder=-2)

    plt.xlim(-4.5, 4.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel('$k_x a$')
    plt.ylabel('$k_y a$')
    plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)

    plt.savefig(r'c:\users\gugli\desktop\tesi\figure\meshgrid.jpeg', dpi = my_dpi*5)

    plt.show()