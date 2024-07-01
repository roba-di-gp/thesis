import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from matplotlib import font_manager
from pathlib import Path

matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)

plt.rcParams['font.family'] = 'Palatino Linotype'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Palatino Linotype'
plt.rcParams['mathtext.it'] = 'Palatino Linotype:italic'
plt.rcParams['mathtext.bf'] = 'Palatino Linotype:bold:italic'

from numpy.linalg import eigh, inv
import os
os.chdir('c:/users/gugli/desktop/tesi/codice')

#draws an hexagon
def draw_hexagon(ax, vertices, lw):
    # Create a Polygon patch with the specified vertices
    hexagon = patches.Polygon(vertices, closed=True, edgecolor='k', facecolor='none',alpha=1, linewidth=lw, zorder= -3)

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

#plot NN, NNN, NNNN vectors
my_dpi = 96
fig, ax = plt.subplots(figsize=(450/my_dpi, 450/my_dpi), dpi=my_dpi)
plt.axis('off')
ax.set_aspect('equal', 'box')

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
#ax.set_xlim(-3.6, 2.8)
#ax.set_ylim(-3.2, 3.2)

lw = 1.2
my_ms = 12
#NN plot
for i in range(len(d0)):
    plt.plot(d0[i][0], d0[i][1], 'r.',zorder=4, markersize = my_ms)
    plt.plot(d0[i][0], d0[i][1], 'k.',zorder=3, markersize = 1.5*my_ms)
    plt.plot([0,d0[i][0]], [0,d0[i][1]], 'r-', linewidth=lw,zorder=2)

plt.plot([0,d0[0][0]], [0,d0[0][1]], 'r-', linewidth=lw,zorder=2, label='NN')

#NNN plot
for i in range(len(d1)):
    plt.plot(d1[i][0], d1[i][1], 'b.',zorder=4, markersize = my_ms)
    plt.plot(d1[i][0], d1[i][1], 'k.',zorder=3, markersize = 1.5*my_ms)
    plt.plot([0,d1[i][0]], [0,d1[i][1]], 'b--', linewidth=lw,zorder=2)

plt.plot([0,d1[0][0]], [0,d1[0][1]], 'b--', linewidth=lw, zorder=2, label='NNN')

#NNNN plot
for i in range(len(d2)):
    plt.plot([0,d2[i][0]], [0,d2[i][1]], 'r.',zorder=4, markersize = my_ms)
    plt.plot([0,d2[i][0]], [0,d2[i][1]], 'k.',zorder=3, markersize =1.5*my_ms)
    plt.plot([0,d2[i][0]], [0,d2[i][1]], 'r:', linewidth=lw, zorder=2)

plt.plot([0,d2[0][0]], [0,d2[0][1]], 'r:', linewidth=lw, zorder=1, label='NNNN')

plt.plot(0,0, 'b.', markersize = my_ms, zorder=4)

plt.legend(shadow=True, loc = 'upper left', prop={'size': 12})
plt.xlabel('$x/a$')
plt.ylabel('$y/a$')
plt.subplots_adjust(left=0., right=1., bottom = 0., top = 1.)

plt.savefig('c:/users/gugli/desktop/tesi/figure/neighbors.png', dpi = my_dpi*5)
plt.show()

##lattice plot

#plot NN, NNN, NNNN vectors
my_dpi = 96
fig, ax = plt.subplots(figsize=(450/my_dpi, 450/my_dpi), dpi=my_dpi, frameon=False)
plt.axis('off')
ax.set_aspect('equal', 'box')
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
#draw_hexagon(ax, v2, lw)
draw_hexagon(ax, v3, lw)
draw_hexagon(ax, v4, lw)
#draw_hexagon(ax, v5, lw)
#draw_hexagon(ax, v6, lw)

#set axis limits
#ax.set_xlim(-3.6, 2.8)
#ax.set_ylim(-3.2, 3.2)

my_ms = 12
plt.plot(0,0,'b.',zorder=3 , markersize = my_ms)
plt.plot(0,0,'k.',zorder=2 , markersize = 1.5*my_ms)
#NN plot
for i in range(len(d0)):
    plt.plot(d0[i][0], d0[i][1], 'r.',zorder=1, markersize = my_ms)
    plt.plot(d0[i][0], d0[i][1], 'k.',zorder=0, markersize = 1.5*my_ms)
#NNN plot
for i in range(len(d1)):
    plt.plot(d1[i][0], d1[i][1], 'b.',zorder=1 , markersize = my_ms)
    plt.plot(d1[i][0], d1[i][1], 'k.',zorder=0 , markersize = 1.5*my_ms)
#NNNN plot
for i in range(len(d2)):
    plt.plot(d2[i][0], d2[i][1], 'r.',zorder=1, markersize = my_ms)
    plt.plot(d2[i][0], d2[i][1], 'k.',zorder=0, markersize = 1.5*my_ms)

plt.arrow(0,0,3/2*0.9, np.sqrt(3)/2*0.9, linestyle = '-', linewidth= 1.5,capstyle ='round',head_width=0.1, fill = True, zorder = 2, facecolor = 'k')
plt.arrow(0,0,3/2*0.9, -np.sqrt(3)/2*0.9, linestyle = '-', linewidth= 1.5,capstyle ='round',head_width=0.1, fill = True, zorder = 2, facecolor = 'k')
plt.arrow(0,0,1/2*0.8, np.sqrt(3)/2*0.8, linestyle = '-', linewidth= 1.5,capstyle ='round',head_width=0.1, fill = True, zorder = 2, facecolor = 'k')
plt.text(0.8,0.25, r'$\mathbf{a}_1$', fontsize = 14)
plt.text(0.8,-0.35, r'$\mathbf{a}_2$', fontsize = 14)
plt.text(-0.15,0.4, r'$\mathbf{\tau}_{\mathrm{b}}$', fontsize = 14)
plt.text(1/2-0.02, np.sqrt(3)/2+0.15, r'B', color='r', fontsize = 16)
plt.text(3/2-0.09, np.sqrt(3)/2+0.15, r'A', color='b', fontsize = 16)

#plt.legend(shadow=True, loc = 'upper left', prop={'size': 10})
plt.xlabel('$x/a$')
plt.ylabel('$y/a$')
#plt.subplots_adjust(left=0.15, right=0.85, bottom = 0.15, top = 0.85)

plt.savefig('c:/users/gugli/desktop/tesi/figure/lattice.png', dpi = my_dpi*5)
plt.show()

##FBZ plot

b1 = np.array([2*np.pi/3, 2*np.pi/np.sqrt(3)])
b2 = np.array([2*np.pi/3, -2*np.pi/np.sqrt(3)])
my_dpi = 96
fig, ax = plt.subplots(figsize=(450/my_dpi, 450/my_dpi), dpi=my_dpi)
plt.axis('off')
ax.set_aspect('equal', 'box')

vertices1 = [
    (0, 4*np.pi/3/np.sqrt(3)),
    (2*np.pi/3, 2*np.pi/3/np.sqrt(3)),
    (2*np.pi/3, -2*np.pi/3/np.sqrt(3)),
    (0, -4*np.pi/3/np.sqrt(3)),
    (-2*np.pi/3, -2*np.pi/3/np.sqrt(3)),
    (-2*np.pi/3, 2*np.pi/3/np.sqrt(3))
]
lw = 0.7
my_fs = 14
draw_hexagon(ax, vertices1, lw)
plt.text(-0.55,-0.45,'$\Gamma$',fontsize = my_fs)
plt.text(2.35,1.1,'K',fontsize = my_fs)
plt.text(2.35,-1.35,'K\'',fontsize = my_fs)
plt.text(2.35,0.15,'M',fontsize = my_fs)

my_ms = 10
plt.plot(b1[0], b1[1]/3, marker = '.', color = 'r', markersize=my_ms, zorder=3, linewidth=0, alpha=1)
plt.plot(b1[0], -b1[1]/3, marker = '.', color = 'r', markersize=my_ms, zorder=3, linewidth=0, alpha=1)
plt.plot(b1[0], 0, marker = '.', color = 'r', markersize=my_ms, zorder=3, linewidth=0, alpha=1)
plt.plot(0, 0, marker = '.', color = 'r', markersize=my_ms, zorder=3, linewidth=0, alpha=1)
plt.plot(b1[0], b1[1]/3, marker = '.', color = 'k', markersize=my_ms*1.5, zorder=2, linewidth=0, alpha=1)
plt.plot(b1[0], -b1[1]/3, marker = '.', color = 'k', markersize=my_ms*1.5, zorder=2, linewidth=0, alpha=1)
plt.plot(b1[0], 0, marker = '.', color = 'k', markersize=my_ms*1.5, zorder=2, linewidth=0, alpha=1)
plt.plot(0, 0, marker = '.', color = 'k', markersize=my_ms*1.5, zorder=2, linewidth=0, alpha=1)

plt.arrow(0,0,0.9*b2[0], 0.91*b2[1], linestyle = '-', linewidth= 1.5,capstyle ='round',head_width=0.2, fill = True, zorder = 2, facecolor = 'k')
plt.arrow(0,0,0.9*b1[0], 0.91*b1[1], linestyle = '-', linewidth= 1.5,capstyle ='round',head_width=0.2, fill = True, zorder = 2, facecolor = 'k')
plt.arrow(0,0,0.9*b1[0], 0.91*b1[1], linestyle = '-', linewidth= 1.5,capstyle ='round',head_width=0.2, fill = True, zorder = 2, facecolor = 'k')
plt.plot([b1[0],2*b1[0]],[b1[1],0],'k--', linewidth=lw, zorder=-2, alpha=1)
plt.plot([b2[0],2*b2[0]],[b2[1],0],'k--', linewidth=lw, zorder=-2, alpha=1)

plt.arrow(0,-4,0,8, linestyle = '-', linewidth= 0.5,capstyle ='round',head_width=0.12, fill = True, zorder = 2, facecolor = 'k')
plt.arrow(-3,0,8,0, linestyle = '-', linewidth= 0.5,capstyle ='round',head_width=0.12, fill = True, zorder = 2, facecolor = 'k')

plt.text(4.5,-0.5, r'$k_x$', fontsize = 14, zorder = -4)
plt.text(-0.65,3.5, r'$k_y$', fontsize = 14, zorder = -4)

plt.text(0.9,2.9, r'$\mathbf{b}_{1}$', fontsize = 14)
plt.text(0.9,-3, r'$\mathbf{b}_{2}$', fontsize = 14)

plt.xlim(-3.,5.2)
plt.ylim(-4.,4.2)

plt.savefig('c:/users/gugli/desktop/tesi/figure/reciprocal_lattice.png', dpi = my_dpi*5)
plt.show()

##high-symmetry contour
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
fig0, ax0 = plt.subplots(figsize=(450/my_dpi, 450/my_dpi), dpi=my_dpi)
plt.axis('off')
ax0.set_aspect('equal', 'box')
vertices1 = [
    (0, 4*np.pi/3/np.sqrt(3)),
    (2*np.pi/3, 2*np.pi/3/np.sqrt(3)),
    (2*np.pi/3, -2*np.pi/3/np.sqrt(3)),
    (0, -4*np.pi/3/np.sqrt(3)),
    (-2*np.pi/3, -2*np.pi/3/np.sqrt(3)),
    (-2*np.pi/3, 2*np.pi/3/np.sqrt(3))
]

lw = 1.5
lw1 = 3
my_ms = 12
draw_hexagon(ax0, vertices1, lw)

plt.plot(0,0,'k.', ms = my_ms*1.5, zorder = 3)
plt.plot(0,0,'r.', ms = my_ms,zorder =4)
plt.plot(2*np.pi/3,2*np.pi/3/np.sqrt(3),'k.', ms = my_ms*1.5, zorder = 3)
plt.plot(2*np.pi/3,2*np.pi/3/np.sqrt(3),'r.', ms = my_ms,zorder =4)
plt.plot(2*np.pi/3,0,'k.', ms = my_ms*1.5, zorder = 3)
plt.plot(2*np.pi/3,0,'r.', ms = my_ms,zorder =4)

plt.plot(kx1, ky1,'r-', linewidth= lw1, zorder = 2)
plt.arrow(kx1[0],ky1[0], 1, 1/np.sqrt(3), head_width=0.18, facecolor= 'r', edgecolor = 'r', zorder = 2)
plt.plot(kx2, ky2,'r-', linewidth= lw1, zorder = 2)
plt.arrow(kx2[0],ky2[0], 0., -0.5, head_width=0.18, facecolor= 'r', edgecolor = 'r', zorder = 2)
plt.plot(kx3, ky3,'r-', linewidth= lw1, zorder = 2)
plt.arrow(kx3[0],ky3[0], -0.8, 0., head_width=0.18, facecolor= 'r', edgecolor = 'r', zorder = 2)
plt.text(-0.45,-0.1,'$\Gamma$', fontsize = 20)
plt.text(2.25,1.1,'K', fontsize = 20)
plt.text(2.25,-0.1,'M', fontsize = 20)


#plt.grid(True, linestyle=':', alpha=0.5)
lim = 2.7
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.xlabel('$k_x a$')
plt.ylabel('$k_y a$')
#plt.subplots_adjust(left=0.15, right=0.85, bottom = 0.15, top = 0.85)

plt.savefig('c:/users/gugli/desktop/tesi/figure/contour.png', dpi = my_dpi*5)
plt.show()

##torma plot
from matplotlib.patches import Circle, Ellipse
from matplotlib import colors



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

my_dpi = 96
fig, ax = plt.subplots(figsize=(600/my_dpi, 300/my_dpi), dpi=my_dpi)
plt.axis('off')
ax.set_aspect('equal', 'datalim')

plt.arrow(5,45,25,0 ,head_width=1, facecolor = 'k')
plt.arrow(5,45,0,20 ,head_width=1, facecolor = 'k')
plt.arrow(70,45,25,0 ,head_width=1, facecolor = 'k')
plt.arrow(70,45,0,20 ,head_width=1, facecolor = 'k')

xx = np.linspace(7, 29, 100)
plt.plot(xx, 3*np.cos(2*np.pi/22*(xx-7))+60, 'r', zorder = 1)
plt.plot(xx, -3*np.cos(2*np.pi/22*(xx-7))+50, 'b', zorder = 1)

xx0 = np.linspace(72, 94, 100)
cc = plt.get_cmap('hsv_r')
cc_r = plt.get_cmap('hsv')
cmap_r = truncate_colormap(cc_r, 0.85, 1.)
cmap = truncate_colormap(cc, 0., 0.15)
plt.scatter(xx0[0:50], 3*np.cos(2*np.pi/22*(xx0[0:50]-72))+60, c = xx0[0:50], cmap = cmap, s = 1.5, zorder = 1)
plt.scatter(xx0[50:99], 3*np.cos(2*np.pi/22*(xx0[50:99]-72))+60, c = xx0[50:99], cmap = cmap_r, s = 1.5, zorder = 1)


cc = plt.get_cmap('hsv')
cc_r = plt.get_cmap('hsv_r')
cmap = truncate_colormap(cc, 0.7, 0.85)
cmap_r = truncate_colormap(cc_r, 0.15, 0.3)
plt.scatter(xx0[0:50], -3*np.cos(2*np.pi/22*(xx0[0:50]-72))+50, c = xx0[0:50], cmap = cmap, s = 1.5, zorder = 1)
plt.scatter(xx0[50:99], -3*np.cos(2*np.pi/22*(xx0[50:99]-72))+50, c = xx0[50:99], cmap = cmap_r, s = 1.5, zorder = 1)

yc = 45
circle = Circle((50, yc), 14, facecolor = 'w', edgecolor = 'k', lw = 1.5, zorder = 1)
ax.add_patch(circle)
ellipse = Ellipse((50, yc), 28, 7, facecolor = 'w', edgecolor = 'k', lw = 1, linestyle = '--', zorder = 1)
ax.add_patch(ellipse)

plt.text(48, yc+14+2.5, r'$\left\vert{\alpha}_{}^{^{}}\right\rangle$')
plt.text(48, yc-14-4, r'$\left\vert{\beta}_{}^{^{}}\right\rangle$')
plt.text(1, 64, r'$E$')
plt.text(66, 64, r'$E$')
plt.text(27, 41, r'$k$')
plt.text(92, 41, r'$k$')

plt.text(13, 35, '$g = 0$', bbox=dict(facecolor='none', edgecolor='k', pad=5.0))
plt.text(78, 35, r'$g \neq 0$', bbox=dict(facecolor='none', edgecolor='k', pad=5.0))

plt.text(15, 64, r'$\left\vert u_{\mathbf{k}} \right\rangle$')
plt.text(17, 70, r'$\left\vert u_{\mathbf{k}+\mathrm{d}\mathbf{k}}\right\rangle$')
plt.arrow(17.8,62.2,0,-2.5 ,head_width=1, facecolor = 'k')
plt.arrow(21.7,68.2,0,-6 ,head_width=1, facecolor = 'k')

my_ms = 8
plt.plot(50, yc+14, 'k.', ms = my_ms*1.5, zorder = 3)
plt.plot(50, yc+14, 'r.', ms = my_ms, zorder = 4)
plt.plot(50, yc-14, 'k.', ms = my_ms*1.5, zorder = 3)
plt.plot(50, yc-14, 'b.', ms = my_ms, zorder = 4)
plt.plot(61, yc+np.sqrt(75), 'k.', ms = my_ms*1.5, zorder = 3)
plt.plot(61, yc+np.sqrt(75), '.', color = 'deeppink', ms = my_ms, zorder = 3)
plt.plot(64, yc, 'k.', ms = my_ms*1.5, zorder = 3)
plt.plot(64, yc, '.',color = 'magenta',  ms = my_ms, zorder = 3)


my_ms = 6
plt.plot(xx[49], 3*np.cos(2*np.pi/22*(xx[49]-7))+60, 'k.', ms = my_ms, zorder = 3)
plt.plot(xx[67], 3*np.cos(2*np.pi/22*(xx[67]-7))+60, 'k.', ms = my_ms, zorder = 3)
plt.plot([xx[49], 50], [3*np.cos(2*np.pi/22*(xx[49]-7))+60, yc+14],'k-',  zorder = 2, alpha = 0.4, linewidth=1)
plt.plot([xx[67], 50], [3*np.cos(2*np.pi/22*(xx[67]-7))+60, yc+14],'k-',  zorder = 2, alpha = 0.4, linewidth=1)

plt.plot(xx0[20], 3*np.cos(2*np.pi/22*(xx0[20]-72))+60, 'k.', ms = my_ms, zorder = 3)
plt.plot(xx0[5], 3*np.cos(2*np.pi/22*(xx0[5]-72))+60, 'k.', ms = my_ms, zorder = 3)
plt.plot(xx0[50], 3*np.cos(2*np.pi/22*(xx0[50]-72))+60, 'k.', ms = my_ms, zorder = 3)

plt.plot([xx0[20], 61], [3*np.cos(2*np.pi/22*(xx0[20]-72))+60, yc+np.sqrt(75)],'k-',  zorder = 2, alpha = 0.4, linewidth=1)
plt.plot([xx0[5], 50], [3*np.cos(2*np.pi/22*(xx0[5]-72))+60, yc+14],'k-',  zorder = 2, alpha = 0.4, linewidth=1)
plt.plot([xx0[50], 64], [3*np.cos(2*np.pi/22*(xx0[50]-72))+60, yc], 'k-',  zorder = 2, alpha = 0.4, linewidth=1)

xx1 = np.linspace(50, 61, 500)
cc = plt.get_cmap('hsv_r')
cmap = truncate_colormap(cc, 0, .07)
plt.scatter(xx1, yc +np.sqrt(14**2- (xx1-50)**2), c = xx1, cmap = cmap, zorder = 3, s = 1)
xx1 = np.linspace(61, 64, 500)
cc = plt.get_cmap('hsv_r')
cmap = truncate_colormap(cc, .07, .15)
plt.scatter(xx1, yc +np.sqrt(14**2- (xx1-50)**2), c = xx1, cmap = cmap, zorder = 3, s = 1)

plt.subplots_adjust(left=0.05, right=1, bottom = 0.05, top=0.9)
plt.savefig('c:/users/gugli/desktop/tesi/figure/torma.png', dpi = my_dpi*5)

plt.show()
