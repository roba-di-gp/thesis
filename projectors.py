import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
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

##path plot
#draws an hexagon
def draw_hexagon(ax, vertices, lw):
    # Create a Polygon patch with the specified vertices
    hexagon = patches.Polygon(vertices, closed=True, edgecolor='k', facecolor='none',alpha=1, linewidth=lw, zorder = -5)

    # Add the hexagon patch to the Axes
    ax.add_patch(hexagon)

my_dpi = 96
fig, ax = plt.subplots(figsize=(400/my_dpi, 400/my_dpi), dpi=my_dpi)
lw = 0.8
lw1 = 2.

vertices1 = [
    (0, 4*np.pi/3/np.sqrt(3)),
    (2*np.pi/3, 2*np.pi/3/np.sqrt(3)),
    (2*np.pi/3, -2*np.pi/3/np.sqrt(3)),
    (0, -4*np.pi/3/np.sqrt(3)),
    (-2*np.pi/3, -2*np.pi/3/np.sqrt(3)),
    (-2*np.pi/3, 2*np.pi/3/np.sqrt(3))
]

ext = 0.5
k1x = np.linspace(2*np.pi/3 - ext, 2*np.pi/3 + ext, 1000)
k1y = np.ones(len(k1x))*2*np.pi/3/np.sqrt(3)
k1 = np.array([k1x, k1y])
k1 = np.transpose(k1)
k2y = np.linspace(2*np.pi/3/np.sqrt(3) - ext, 2*np.pi/3/np.sqrt(3) + ext,100)
k2x = np.ones(len(k2y))*2*np.pi/3
k2 = np.array([k2x, k2y])
k2 = np.transpose(k2)

plt.plot(k1x, k1y, 'r-', linewidth = lw1, zorder = -1)
plt.arrow(2*np.pi/3+0.1, 2*np.pi/3/np.sqrt(3),0.3 , 0, head_width=0.18, facecolor= 'r', edgecolor = 'r', zorder = 1)
#plt.plot(k2x, k2y, 'b-', linewidth = lw1, zorder = -1)
#plt.arrow(2*np.pi/3, 2*np.pi/3/np.sqrt(3),0 , 0.4, head_width=0.18, facecolor= 'b', edgecolor = 'b', zorder = 1)


draw_hexagon(ax, vertices1, lw)

plt.xlabel('$k_x a$')
plt.ylabel('$k_y a$')
plt.subplots_adjust(left=0.15, right=0.85, bottom = 0.15, top = 0.85)
plt.xlim(-3.7, 3.7)
plt.ylim(-3.7, 3.7)

plt.show()

##projector - plot across the valley
my_dpi = 96
fig1, ax1 = plt.subplots(figsize=(400/my_dpi, 400/my_dpi), dpi=my_dpi)
lw = 0.8
lw1 = 2.

ext = 0.03
k1y = np.linspace(4*np.pi/3/np.sqrt(3)-ext, 4*np.pi/3/np.sqrt(3)+ext, 200)
k1x = np.zeros(len(k1y))
k1 = np.array([k1x, k1y])
k1 = np.transpose(k1)

gap = 0.02
pp= np.load('c:/users/gugli/desktop/tesi/data/bande_bestfit.npy')
p1c = []
p1v = []
for k in k1:
    null, null, pp1 = fun.EUP(k, pp, gap)
    p1v.append(pp1[0,0,1].real)
    p1c.append(pp1[1,0,1].real)

p1v = np.array(p1v)
p1c = np.array(p1c)
plt.plot(k1y, p1v, 'k.', markersize = 8)
plt.plot(k1y, p1v, 'b.', markersize = 3)
plt.plot(k1y, p1c, 'k.', markersize = 8)
plt.plot(k1y, p1c, 'r.', markersize = 3)
plt.xlabel('$k_y a$')
#plt.ylabel('Re $[P_{\mathrm{v}}(\mathbf{k})]_{AB}$')
plt.xticks([4*np.pi/3/np.sqrt(3)-ext, 4*np.pi/3/np.sqrt(3), 4*np.pi/3/np.sqrt(3)+ext], ['$\mathbf{K} - 0.03\hat{\mathbf{k}}_y$','$\mathbf{K}$', '$\mathbf{K} + 0.03\hat{\mathbf{k}}_y$'])
plt.title('$\Delta = 20$ meV')

plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
plt.xlim(4*np.pi/3/np.sqrt(3)- ext, 4*np.pi/3/np.sqrt(3)+ ext)
plt.ylim(-0.7, 0.7)
plt.grid(axis='x', linestyle = '--', alpha = 0.8, zorder = -5, linewidth=1.5)

blue_patch = matplotlib.patches.Patch(facecolor='r',edgecolor='k', label = 'Re$[P_{\mathrm{+}}(\mathbf{k})]_{AB}$')
red_patch = matplotlib.patches.Patch(facecolor='b',edgecolor='k', label = 'Re$[P_{\mathrm{-}}(\mathbf{k})]_{AB}$')
plt.legend(handles=[blue_patch, red_patch], shadow=True, loc = [0.7,0.4], prop={'size': 11})


plt.savefig('c:/users/gugli/desktop/tesi/figure/proj_y_20mev.png', dpi = my_dpi*5)

plt.show()

##derivative - AB - plot across the valley
my_dpi = 96
fig1, ax1 = plt.subplots(figsize=(440/my_dpi, 400/my_dpi), dpi=my_dpi)
lw = 0.8
lw1 = 2.

ext = 0.01
k1y = 2*np.pi/3/np.sqrt(3)*np.ones(400)
k1x = np.linspace(2*np.pi/3-ext, 2*np.pi/3+ext, len(k1y))
k1 = np.array([k1x, k1y])
k1 = np.transpose(k1)

dky = k1x[1]-k1x[0]
dk = np.array([dky, dky])

gap = 0.02
pp= np.load('c:/users/gugli/desktop/tesi/data/bande_bestfit.npy')
p1c = []
p1v = []
for k in k1:
    null, dpp1 = fun.dEdP(k, pp, dk, gap)
    p1v.append(dpp1[0,0,0,0].real)
    p1c.append(dpp1[0,1,0,0].real)

p1v = np.array(p1v)
p1c = np.array(p1c)
plt.plot(k1x, p1v, 'k.', markersize = 8)
plt.plot(k1x, p1v, 'b.', markersize = 3)
plt.plot(k1x, p1c, 'k.', markersize = 8)
plt.plot(k1x, p1c, 'r.', markersize = 3)
plt.xlabel('$k_x a$')
plt.ylabel('[nm]')
#plt.ylabel('Re $[P_{\mathrm{v}}(\mathbf{k})]_{AB}$')
plt.xticks([2*np.pi/3-ext, 2*np.pi/3, 2*np.pi/3+ext], ['$\mathbf{K} - 0.01\hat{\mathbf{k}}_x$','$\mathbf{K}$', '$\mathbf{K} + 0.01\hat{\mathbf{k}}_x$'])
plt.title('$\Delta = 20$ meV')

plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
plt.xlim(2*np.pi/3- ext, 2*np.pi/3+ ext)
#plt.ylim(-0.7, 0.7)
plt.grid(axis='x', linestyle = '--', alpha = 0.8, zorder = -5, linewidth=1.5)


blue_patch = matplotlib.patches.Patch(facecolor='r',edgecolor='k', label = '$\partial_{x}$Re$[P_{\mathrm{+}}(\mathbf{k})]_{AA}$')
red_patch = matplotlib.patches.Patch(facecolor='b',edgecolor='k', label = '$\partial_{x}$Re$[P_{\mathrm{-}}(\mathbf{k})]_{AA}$')
plt.legend(handles=[blue_patch, red_patch], shadow=True, loc = [0.65,0.6], prop={'size': 11})


plt.savefig('c:/users/gugli/desktop/tesi/figure/dpdx_AA_20mev.png', dpi = my_dpi*5)

plt.show()

##2nd derivative dydy - AB - plot across the valley
my_dpi = 96
fig1, ax1 = plt.subplots(figsize=(440/my_dpi, 400/my_dpi), dpi=my_dpi)
lw = 0.8
lw1 = 2.

ext = 0.01
k1y = np.linspace(4*np.pi/3/np.sqrt(3)-ext, 4*np.pi/3/np.sqrt(3)+ext, 200)
k1x = np.zeros(len(k1y))
k1 = np.array([k1x, k1y])
k1 = np.transpose(k1)

dky = k1y[1]-k1y[0]
dk = np.array([dky, dky])

gap = 0.02
pp= np.load('c:/users/gugli/desktop/tesi/data/bande_bestfit.npy')
p1c = []
p1v = []
for k in k1:
    null, ddpp1 = fun.ddEddP(k, pp, dk, gap)
    p1v.append(ddpp1[1,1,0,0,1].real)
    p1c.append(ddpp1[1,1,1,0,1].real)

p1v = np.array(p1v)
p1c = np.array(p1c)
plt.plot(k1y, p1v, 'k.', markersize = 8)
plt.plot(k1y, p1v, 'b.', markersize = 3)
plt.plot(k1y, p1c, 'k.', markersize = 8)
plt.plot(k1y, p1c, 'r.', markersize = 3)
plt.xlabel('$k_y a$')
plt.ylabel('[nm$^2$]')
#plt.ylabel('Re $[P_{\mathrm{v}}(\mathbf{k})]_{AB}$')
plt.xticks([4*np.pi/3/np.sqrt(3)-ext, 4*np.pi/3/np.sqrt(3), 4*np.pi/3/np.sqrt(3)+ext], ['$\mathbf{K} - 0.03\hat{\mathbf{k}}_y$','$\mathbf{K}$', '$\mathbf{K} + 0.03\hat{\mathbf{k}}_y$'])
plt.title('$\Delta = 20$ meV')


plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
plt.xlim(4*np.pi/3/np.sqrt(3)- ext, 4*np.pi/3/np.sqrt(3)+ ext)
#plt.ylim(-0.7, 0.7)
plt.grid(axis='x', linestyle = '--', alpha = 0.8, zorder = -5, linewidth=1.5)


blue_patch = matplotlib.patches.Patch(facecolor='r',edgecolor='k', label = '$\partial^{2}_{yy}$Re$[P_{\mathrm{+}}(\mathbf{k})]_{AB}$')
red_patch = matplotlib.patches.Patch(facecolor='b',edgecolor='k', label = '$\partial^{2}_{yy}$Re$[P_{\mathrm{-}}(\mathbf{k})]_{AB}$')
plt.legend(handles=[blue_patch, red_patch], shadow=True, loc = [0.65,0.6], prop={'size': 11})


plt.savefig('c:/users/gugli/desktop/tesi/figure/ddpdydy_AB_20mev.png', dpi = my_dpi*5)

plt.show()

##2nd derivative dydy - AA - plot across the valley

my_dpi = 96
fig1, ax1 = plt.subplots(figsize=(440/my_dpi, 400/my_dpi), dpi=my_dpi)
lw = 0.8
lw1 = 2.

ext = 0.01
k1y = np.linspace(4*np.pi/3/np.sqrt(3)-ext, 4*np.pi/3/np.sqrt(3)+ext, 200)
k1x = np.zeros(len(k1y))
k1 = np.array([k1x, k1y])
k1 = np.transpose(k1)

dky = (k1y[1]-k1y[0])
k = np.array([dky, dky])

gap = 0.02
pp= np.load('c:/users/gugli/desktop/tesi/data/bande_bestfit.npy')
p1c = []
p1v = []
for k in k1:
    null, ddpp1 = fun.ddEddP(k, pp, dk, gap)
    p1v.append(ddpp1[1,1,0,0,0].real)
    p1c.append(ddpp1[1,1,1,0,0].real)

p1v = np.array(p1v)
p1c = np.array(p1c)
plt.plot(k1y, p1v, 'k.', markersize = 8)
plt.plot(k1y, p1v, 'b.', markersize = 3)
plt.plot(k1y, p1c, 'k.', markersize = 8)
plt.plot(k1y, p1c, 'r.', markersize = 3)
plt.xlabel('$k_y a$')
plt.ylabel('[nm$^2$]')
#plt.ylabel('Re $[P_{\mathrm{v}}(\mathbf{k})]_{AB}$')
plt.xticks([4*np.pi/3/np.sqrt(3)-ext, 4*np.pi/3/np.sqrt(3), 4*np.pi/3/np.sqrt(3)+ext], ['$\mathbf{K} - 0.03\hat{\mathbf{k}}_y$','$\mathbf{K}$', '$\mathbf{K} + 0.03\hat{\mathbf{k}}_y$'])
plt.title('$\Delta = 20$ meV')


plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
plt.xlim(4*np.pi/3/np.sqrt(3)- ext, 4*np.pi/3/np.sqrt(3)+ ext)
#plt.ylim(-0.7, 0.7)
plt.grid(axis='x', linestyle = '--', alpha = 0.8, zorder = -5, linewidth=1.5)


blue_patch = matplotlib.patches.Patch(facecolor='r',edgecolor='k', label = '$\partial^{2}_{yy}$Re$[P_{\mathrm{+}}(\mathbf{k})]_{AA}$')
red_patch = matplotlib.patches.Patch(facecolor='b',edgecolor='k', label = '$\partial^{2}_{yy}$Re$[P_{\mathrm{-}}(\mathbf{k})]_{AA}$')
plt.legend(handles=[blue_patch, red_patch], shadow=True, loc = [0.65,0.7], prop={'size': 11})


plt.savefig('c:/users/gugli/desktop/tesi/figure/ddpdydy_AA_20mev.png', dpi = my_dpi*5)

plt.show()

##colorplots - Re pAB

cmap = plt.get_cmap('jet_r')
my_ms = 6

K = [-2*np.pi/3, 2*np.pi/3/np.sqrt(3)]

lim = 3.2
b1 = np.array([lim,0])
b2 = np.array([0,lim])

#build the grid on the PC
N = 36
kpc = fun.ext_fbz_meshgrid(N,1)

#NNNN
gap = 0.0 #[eV]
#calculate the condband projector
repc = []
pp= np.load('c:/users/gugli/desktop/tesi/data/bande_bestfit.npy')
#pp= [0,pp[1],0,0,0,0,0]
for k in kpc:
    null, u, p1 = fun.EUP(k, pp, gap)
    repc.append(p1[1,0,1].real)

repc = np.array(repc)

my_dpi = 96
fig, ax= plt.subplots(figsize=(2*300/my_dpi, 1*300/my_dpi), dpi=my_dpi)
my_lw = 1.2
ax.set_aspect('equal', 'box')

sc = plt.scatter(kpc[:,0], kpc[:,1], c = repc, cmap = cmap, s = my_ms, marker = 'h')
plt.colorbar(sc, fraction = 0.045)
plt.subplots_adjust(left=0.2, right=0.9, bottom = 0.2, top = 0.9)
if gap > 0:
    plt.title('$\mathrm{Re} P^{AB}_{+} (\mathbf{k})$ @$\Delta = %.0f$ meV'%(gap*1000))
if gap == 0:
    plt.title('$\mathrm{Re} P^{AB}_{+} (\mathbf{k})$ @$\Delta = 0$')

ax.set_xlabel('$k_x a$')
ax.set_ylabel('$k_y a$')
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)

#rebuild the grid for the zoom
limz = 0.025
b1 = np.array([limz,0])
b2 = np.array([0,limz])

my_ms = 8
N = 35
kpcz = fun.ext_fbz_meshgrid(N, 0.012, K)

#NNNN
#calculate the condband projector
repcz = []
for k in kpcz:
    null, u, p1 = fun.EUP(k, pp, gap)
    repcz.append(p1[1,0,1].real)

repcz = np.array(repcz)

axins = ax.inset_axes([-3.2*lim, -0.7*lim, 1.4*lim, 1.4*lim], transform=ax.transData)
axins.scatter(kpcz[:,0], kpcz[:,1], c = repcz, cmap = cmap, s = my_ms, marker = 'h')
axins.set_xlim(K[0]-limz, K[0] +limz)
axins.set_ylim(K[1]-limz, K[1]+limz)
# axins.set_xticks([])
# axins.set_yticks([])
# axins.set_xticklabels([])
# axins.set_yticklabels([])
ax.indicate_inset_zoom(axins, edgecolor='k', alpha=0.7)

path = 'c:/users/gugli/desktop/tesi/figure/repAB' + '%.0f'%(gap*1000) + 'meV.png'
plt.savefig(path, dpi = my_dpi*5)

plt.show()

##FSM

my_dpi = 96
fig1, ax1 = plt.subplots(figsize=(500/my_dpi, 300/my_dpi), dpi=my_dpi)
lw = 0.8
lw1 = 2.

ext = 0.01

k1y = np.linspace(4*np.pi/3/np.sqrt(3)-ext, 4*np.pi/3/np.sqrt(3)+ext, 200)
k1x = np.zeros(len(k1y))
k1 = np.array([k1x, k1y])
k1 = np.transpose(k1)
dkx = (k1y[1]-k1y[0])/100
dk = np.array([dkx, dkx])

gap = 0.02
pp= np.load('c:/users/gugli/desktop/tesi/data/bande_bestfit.npy')
pp_nn = [0,pp[1],0,0,0,0,0]
trgc = []
trgc_nn = []
for k in k1:
    null, dp = fun.dEdP(k, pp, dk, gap)
    dpc = dp[:,1,:,:]
    trgc1 = 0.5*np.einsum('ijk, ikj', dpc, dpc)
    trgc.append(trgc1)
    null, dp_nn = fun.dEdP(k, pp_nn, dk, gap)
    dpc_nn = dp_nn[:,1,:,:]
    trgc1_nn = 0.5*np.einsum('ijk, ikj', dpc_nn, dpc_nn)
    trgc_nn.append(trgc1_nn)

trgc = np.array(trgc)
trgc_nn = np.array(trgc_nn)


#theoretical value after regularization (debug, manca 1/2)
trg_th = 1/2*(3*pp[1]/gap*a)**2
dec = 2
plt.plot(k1y[::dec], trgc[::dec]/1e3, 'k.-', markersize = 5, zorder = 2, linewidth=1)
plt.plot(k1y[::dec], trgc[::dec]/1e3, 'magenta',marker='.',linewidth=0,  markersize = 2.5, zorder=3)
plt.plot(k1y, trgc_nn/1e3, 'k.-', markersize = 5, zorder = 0, linewidth=1)
plt.plot(k1y, trgc_nn/1e3, 'cyan',marker='.',linewidth=0,  markersize = 2.5, zorder = 1)
plt.grid(axis = 'x', linestyle = '--', alpha = 0.6, zorder = -5, linewidth=1)

#plt.plot(k1y, 1/(4*(k1y-4*np.pi/3/np.sqrt(3))**2), 'k:', zorder = -2, label= 'NN theory', alpha = 0.3)

plt.plot(k1y, trgc_nn/1e3,'cyan' , marker='.',linewidth=0, markersize = 5, label = 'NN theory', zorder = -3)
plt.plot(k1y[::dec], trgc[::dec]/1e3, 'magenta', marker='.',linewidth=0,  markersize = 5, label = 'NNNN theory', zorder = -3)
plt.plot(k1y, np.ones(len(k1y))*trg_th/1e3, 'k--', zorder = -2,linewidth=1, label='$\mathbf{q} = 0$ limit')
plt.xlabel('$k_y a$')
plt.text(4*np.pi/3/np.sqrt(3)+0.0105,2185/1e3, r'$\frac{\hbar^2}{2m^2v^2}$', size = 14)
plt.text(4*np.pi/3/np.sqrt(3)-0.009,1800/1e3,'$\Delta = 20$ meV', size = 14)
#plt.ylabel('Re $[P_{\mathrm{v}}(\mathbf{k})]_{AB}$')
plt.xticks([4*np.pi/3/np.sqrt(3)-ext, 4*np.pi/3/np.sqrt(3), 4*np.pi/3/np.sqrt(3)+ext], ['$\mathbf{K} - 0.01\hat{\mathbf{k}}_y$','$\mathbf{K}$', '$\mathbf{K} + 0.01\hat{\mathbf{k}}_y$'])


plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.9)
plt.xlim(4*np.pi/3/np.sqrt(3)- ext, 4*np.pi/3/np.sqrt(3)+ ext)
plt.ylim((min(trgc_nn)-0.05*max(trgc_nn))/1e3, max(trgc_nn)*(1+0.1)/1e3)

plt.legend(shadow=True, loc = [0.67,0.4], prop={'size': 10})
plt.ylabel(r'Tr $\left[g_\pm(\mathbf{k})\right]$   [$10^3$ nm$^{2}$]')

#plt.savefig('c:/users/gugli/desktop/tesi/figure/trFSM_20mev.png', dpi = my_dpi*5)

plt.show()

##FSM - colorplots

cmap = plt.get_cmap('jet_r')
my_ms = 6

K = [-2*np.pi/3,2*np.pi/3/np.sqrt(3)]

lim = 3.2
b1 = np.array([lim,0])
b2 = np.array([0,lim])

#build the grid on the PC
N = 35
kpc = fun.ext_fbz_meshgrid(N,1)

kpc = np.array(kpc)

dkx = (kpc[1,1]-kpc[0,1])/100
dk = np.array([dkx, dkx])
#NNNN
gap = 0.03 #[eV]
#calculate the condband projector
trgc = []
pp= np.load('c:/users/gugli/desktop/tesi/data/bande_bestfit.npy')
#pp= [0,pp[1],0,0,0,0,0]
for k in kpc:
    null, dp = fun.dEdP(k, pp, dk, gap)
    dpc = dp[:,1,:,:]
    trgc1 = 0.5*np.einsum('ijk, ikj', dpc, dpc)
    trgc.append(trgc1)

trgc = np.array(trgc)

my_dpi = 96
fig, ax= plt.subplots(figsize=(2*300/my_dpi, 1*300/my_dpi), dpi=my_dpi)
my_lw = 1.2
ax.set_aspect('equal', 'box')

sc = plt.scatter(kpc[:,0], kpc[:,1], c = trgc, cmap = cmap, s = my_ms, marker = 'h')
plt.colorbar(sc, fraction = 0.045, label='[nm$^{2}]$', anchor = (0,0.5))
plt.subplots_adjust(left=0.2, right=0.9, bottom = 0.2, top = 0.9)
if gap > 0:
    plt.title('Tr $g_{+} (\mathbf{k})$ @$\Delta = %.0f$ meV'%(gap*1000))
if gap == 0:
    plt.title('Tr $g_{+} (\mathbf{k})$ @$\Delta = 0$')

ax.set_xlabel('$k_x a$')
ax.set_ylabel('$k_y a$')
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)

#rebuild the grid for the zoom
limz = 0.01
b1 = np.array([limz,0])
b2 = np.array([0,limz])

my_ms = 6
kpcz = fun.ext_fbz_meshgrid(N, 0.004, K)

#NNNN
#calculate the condband projector
trgcz = []
#pp= [0,pp[1],0,0,0,0,0]
for k in kpcz:
    null, dp = fun.dEdP(k, pp, dk, gap)
    dpc = dp[:,1,:,:]
    trgc1 = 0.5*np.einsum('ijk, ikj', dpc, dpc)
    trgcz.append(trgc1)

trgcz = np.array(trgcz)

axins = ax.inset_axes([-3.2*lim, -0.7*lim, 1.4*lim, 1.4*lim], transform=ax.transData)
sc1 = axins.scatter(kpcz[:,0], kpcz[:,1], c = trgcz, cmap = cmap, s = my_ms, marker = 'h')
cb.ax.yaxis.set_ticks_position("left")
cb.ax.yaxis.set_label_position("left")
axins.set_xlim(K[0]-limz, K[0]+limz)
axins.set_ylim(K[1]-limz, K[1]+limz)
# axins.set_xticks([])
# axins.set_yticks([])
# axins.set_xticklabels([])
# axins.set_yticklabels([])
ax.indicate_inset_zoom(axins, edgecolor='k', alpha=0.7)

gapstr = str(gap*1000)
path = 'c:/users/gugli/desktop/tesi/figure/trFSM_cplot'+ '%.0f'%(gap*1000) + 'meV.png'
plt.savefig(path, dpi = my_dpi*5)

plt.show()

##projectors control plot on FBZ grid

cmap = plt.get_cmap('jet')
lim = 3.0
my_ms = 14

fbzmesh = fun.ext_fbz_meshgrid(30)
#differential
dkx = 2*np.pi/3/np.sqrt(len(fbzmesh))/1000
dky = 2*np.pi/np.sqrt(3)/np.sqrt(len(fbzmesh))/1000
my_dk = np.array([dkx, dky])
#NNNN
gap = 0.01 #[eV]
#calculate the derivatives of the condband on the grid
p = []
pp= np.load('c:/users/gugli/desktop/tesi/data/bande_bestfit.npy')
for k in fbzmesh:
    null, null, p1 = fun.EUP(k, pp, gap)
    p.append(p1[1,0,1].real)

p = np.array(p)

my_dpi = 96
fig, ax= plt.subplots(figsize=(450/my_dpi, 450/my_dpi), dpi=my_dpi)
ax.set_aspect('equal', 'box')
my_lw = 1.2

sc = plt.scatter(fbzmesh[:,0],fbzmesh[:,1], c = p, cmap = cmap, s = my_ms, marker = 'h')
plt.colorbar(sc, fraction = 0.045)
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
ax.set_xlabel('$k_x a$')
ax.set_ylabel('$k_y a$')

plt.title('$ \mathrm{Re}  P^{AB}_{+} (\mathbf{k})$ @$\Delta = %.3f$ eV'%(gap))
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
#plt.xticks(ticks, ticklabels)
#plt.yticks(ticks, ticklabels)

plt.savefig('c:/users/gugli/desktop/tesi/figure/pAB_controlplot.png', dpi = my_dpi*5)

plt.show()

##derivatives control plot on FBZ grid

cmap = plt.get_cmap('jet_r')
lim = 2.6
my_ms = 20

fbzmesh1 = fun.fbz_meshgrid(30,1,1)
fbzmesh = []
for i in range(len(fbzmesh1)):
    if abs(fbzmesh1[i,0]) > 0.001:
        fbzmesh.append(fbzmesh1[i])

fbzmesh = np.array(fbzmesh1)

#differential
dkx = 2*np.pi/3/np.sqrt(len(fbzmesh))/1000
dky = 2*np.pi/np.sqrt(3)/np.sqrt(len(fbzmesh))/1000
my_dk = np.array([dkx, dky])
#NNNN
gap = 0.01 #[eV]
#calculate the derivatives of the condband on the grid
dpx = []
dpy = []
pp= np.load('c:/users/gugli/desktop/tesi/data/bande_bestfit.npy')
for k in fbzmesh:
    null, dp = fun.dEdP(k, pp, my_dk, gap)
    dpx.append(dp[0,1,0,1].real)
    dpy.append(dp[1,1,0,1].real)

dpx = np.array(dpx)
dpy = np.array(dpy)

my_dpi = 96
fig3, ax3= plt.subplots(figsize=(1.5*300/my_dpi, 1.5*300/my_dpi), dpi=my_dpi)
ax3.set_aspect('equal', 'box')

sc = plt.scatter(fbzmesh[:,0],fbzmesh[:,1], c = dpx, cmap = cmap, s = my_ms, marker = 'h')
plt.colorbar(sc, fraction = 0.045, label = 'nm')
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
#plt.xticks(ticks, ticklabels)
#plt.yticks(ticks, ticklabels)
plt.title('$\partial_{x}\, \mathrm{Re} P^{AB}_{+} (\mathbf{k})$ @$\Delta = %.3f$ eV'%(gap))
ax3.set_xlabel('$k_x a$')
ax3.set_ylabel('$k_y a$')
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.savefig('c:/users/gugli/desktop/tesi/figure/controlplot_dpdx_fbz.png', dpi = my_dpi*5)

my_dpi = 96
fig4, ax4= plt.subplots(figsize=(1.5*300/my_dpi, 1.5*300/my_dpi), dpi=my_dpi)
ax4.set_aspect('equal', 'box')

sc = plt.scatter(fbzmesh[:,0],fbzmesh[:,1], c = dpy, cmap = cmap, s = my_ms, marker = 'h')
plt.colorbar(sc, fraction = 0.045, label = 'nm')
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
ax4.set_xlabel('$k_x a$')
ax4.set_ylabel('$k_y a$')

plt.title('$\partial_{y}\,  \mathrm{Re}  P^{AB}_{+} (\mathbf{k})$ @$\Delta = %.3f$ eV'%(gap))
#plt.xticks(ticks, ticklabels)
#plt.yticks(ticks, ticklabels)
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)

plt.savefig('c:/users/gugli/desktop/tesi/figure/controlplot_dpdy_fbz.png', dpi = my_dpi*5)

plt.show()

##derivatives control plot on unit cell grid

cmap = plt.get_cmap('jet')
my_ms = 16

b1 = np.array([2*np.pi/3, 2*np.pi/np.sqrt(3)])
b2 = np.array([2*np.pi/3, -2*np.pi/np.sqrt(3)])

xlim = 2*b1[0]
ylim = b1[1]

#build the grid on the PC
N = 28
kpc = []
for i in range(N):
    for j in range(N):
        kpc.append(i/N*b1 + j/N*b2)

kpc = np.array(kpc)

#differential
dkx = 2*np.pi/3/np.sqrt(len(kpc))/1000
dky = 2*np.pi/np.sqrt(3)/np.sqrt(len(kpc))/1000
my_dk = np.array([dkx, dky])
#NNNN
gap = 0.01 #[eV]
#calculate the derivatives of the condband on the grid
dpx = []
dpy = []
pp= np.load('c:/users/gugli/desktop/tesi/data/bande_bestfit.npy')
for k in kpc:
    null, dp = fun.dEdP(k, pp, my_dk, gap)
    dpx.append(dp[0,1,0,1].real)
    dpy.append(dp[1,1,0,1].real)

dpx = np.array(dpx)
dpy = np.array(dpy)

my_dpi = 96
fig3, ax3= plt.subplots(figsize=(1.2*300/my_dpi, 1.2*np.sqrt(3)*300/my_dpi), dpi=my_dpi)
my_lw = 1.2

plt.scatter(kpc[:,0],kpc[:,1], c = dpx, cmap = cmap, s = my_ms, marker = 'd')
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
#plt.xticks(ticks, ticklabels)
#plt.yticks(ticks, ticklabels)
plt.title('$\partial_{x}\, \mathrm{Re} P^{AB}_{+} (\mathbf{k})$ @$\Delta = %.3f$ eV'%(gap))
ax3.set_xlabel('$k_x a$')
ax3.set_ylabel('$k_y a$')
plt.xlim(-0.188, xlim)
plt.ylim(-ylim-0.2, ylim+0.2)
plt.savefig('c:/users/gugli/desktop/tesi/figure/controlplot_dpdx_pc.png', dpi = my_dpi*5)

my_dpi = 96
fig4, ax4= plt.subplots(figsize=(1.2*300/my_dpi, 1.2*np.sqrt(3)*300/my_dpi), dpi=my_dpi)
my_lw = 1.2

plt.scatter(kpc[:,0],kpc[:,1], c = dpy, cmap = cmap, s = my_ms, marker = 'd')
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
ax4.set_xlabel('$k_x a$')
ax4.set_ylabel('$k_y a$')

plt.title('$\partial_{y}\,  \mathrm{Re}  P^{AB}_{+} (\mathbf{k})$ @$\Delta = %.3f$ eV'%(gap))
#plt.xticks(ticks, ticklabels)
#plt.yticks(ticks, ticklabels)
plt.xlim(-0.188, xlim)
plt.ylim(-ylim-0.2, ylim+0.2)

plt.savefig('c:/users/gugli/desktop/tesi/figure/controlplot_dpdy_pc.png', dpi = my_dpi*5)

plt.show()

##derivatives control plot on extendend grid

cmap = plt.get_cmap('jet')
my_ms = 20
lim = 5
#build the grid on the PC
N = 10
kpc = fun.ext_fbz_meshgrid(N)
#differential
dkx = 2*np.pi/3/np.sqrt(len(kpc))/1000
dky = 2*np.pi/np.sqrt(3)/np.sqrt(len(kpc))/1000
my_dk = np.array([dkx, dky])
#NNNN
gap = 0.01 #[eV]
#calculate the derivatives of the condband on the grid
dpx = []
dpy = []
pp= np.load('c:/users/gugli/desktop/tesi/data/bande_bestfit.npy')
for k in kpc:
    null, dp = fun.dEdP(k, pp, my_dk, gap)
    dpx.append(dp[0,1,0,1].real)
    dpy.append(dp[1,1,0,1].real)

dpx = np.array(dpx)
dpy = np.array(dpy)

my_dpi = 96
fig3, ax3= plt.subplots(figsize=(1.5*300/my_dpi, 1.5*300/my_dpi), dpi=my_dpi)
my_lw = 1.2
ax3.set_aspect('equal', 'box')

sc = plt.scatter(kpc[:,0],kpc[:,1], c = dpx, cmap = cmap, s = my_ms, marker = 'h')
plt.colorbar(sc, fraction = 0.045)
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
#plt.xticks(ticks, ticklabels)
#plt.yticks(ticks, ticklabels)
plt.title('$\partial_{x}\, \mathrm{Re} P^{AB}_{+} (\mathbf{k})$ @$\Delta = %.3f$ eV'%(gap))
ax3.set_xlabel('$k_x a$')
ax3.set_ylabel('$k_y a$')
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.savefig('c:/users/gugli/desktop/tesi/figure/controlplot_dpdx_square.png', dpi = my_dpi*5)

my_dpi = 96
fig4, ax4= plt.subplots(figsize=(1.5*300/my_dpi, 1.5*300/my_dpi), dpi=my_dpi)
my_lw = 1.2
ax4.set_aspect('equal', 'box')

cs = plt.scatter(kpc[:,0],kpc[:,1], c = dpy, cmap = cmap, s = my_ms, marker = 'h')
plt.colorbar(sc, fraction = 0.045)
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
ax4.set_xlabel('$k_x a$')
ax4.set_ylabel('$k_y a$')

plt.title('$\partial_{y}\,  \mathrm{Re}  P^{AB}_{+} (\mathbf{k})$ @$\Delta = %.3f$ eV'%(gap))
#plt.xticks(ticks, ticklabels)
#plt.yticks(ticks, ticklabels)
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)

plt.savefig('c:/users/gugli/desktop/tesi/figure/controlplot_dpdy_square.png', dpi = my_dpi*5)

plt.show()

##2nd derivatives control plot on FBZ grid

cmap = plt.get_cmap('jet_r')
lim = 2.6
my_ms = 27

fbzmesh1 = fun.fbz_meshgrid(24)
fbzmesh = []
for i in range(len(fbzmesh1)):
    if abs(fbzmesh1[i,0]) > 0.001:
        fbzmesh.append(fbzmesh1[i])

fbzmesh = np.array(fbzmesh1)

#differential
dkx = 2*np.pi/3/np.sqrt(len(fbzmesh))/1000
dky = 2*np.pi/np.sqrt(3)/np.sqrt(len(fbzmesh))/1000
my_dk = np.array([dkx, dky])
#NNNN
gap = 0.01 #[eV]
#calculate the derivatives of the valband on the grid
dpx = []
dpy = []
pp= np.load('c:/users/gugli/desktop/tesi/data/bande_bestfit.npy')
for k in fbzmesh:
    null, ddp = fun.ddEddP(k, pp, my_dk, gap)
    dpx.append(ddp[0,0,0,0,0].real)
    dpy.append(ddp[0,1,0,0,0].real)

dpx = np.array(dpx)
dpy = np.array(dpy)

my_dpi = 96
fig3, ax3= plt.subplots(figsize=(1.5*300/my_dpi, 1.5*300/my_dpi), dpi=my_dpi)
ax3.set_aspect('equal', 'box')

sc = plt.scatter(fbzmesh[:,0],fbzmesh[:,1], c = dpx, cmap = cmap, s = my_ms, marker = 'h')
plt.colorbar(sc, fraction = 0.045, label = 'nm')
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
#plt.xticks(ticks, ticklabels)
#plt.yticks(ticks, ticklabels)
plt.title('$\partial^{2}_{xx}\, \mathrm{Re} P^{AA}_{-} (\mathbf{k})$ @$\Delta = %.3f$ eV'%(gap))
ax3.set_xlabel('$k_x a$')
ax3.set_ylabel('$k_y a$')
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.savefig('c:/users/gugli/desktop/tesi/figure/controlplot_ddpdxdx_fbz.png', dpi = my_dpi*5)

my_dpi = 96
fig4, ax4= plt.subplots(figsize=(1.5*300/my_dpi, 1.5*300/my_dpi), dpi=my_dpi)
ax4.set_aspect('equal', 'box')

sc = plt.scatter(fbzmesh[:,0],fbzmesh[:,1], c = dpy, cmap = cmap, s = my_ms, marker = 'h')
plt.colorbar(sc, fraction = 0.045, label = 'nm')
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
ax4.set_xlabel('$k_x a$')
ax4.set_ylabel('$k_y a$')

plt.title('$\partial^{2}_{xy}\,  \mathrm{Re}  P^{AA}_{-} (\mathbf{k})$ @$\Delta = %.3f$ eV'%(gap))
#plt.xticks(ticks, ticklabels)
#plt.yticks(ticks, ticklabels)
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)

plt.savefig('c:/users/gugli/desktop/tesi/figure/controlplot_ddpdxdy_fbz.png', dpi = my_dpi*5)

plt.show()

##2nd derivatives control plot on square grid

cmap = plt.get_cmap('jet')
my_ms = 12

lim = 7
b1 = np.array([lim,0])
b2 = np.array([0,lim])

#build the grid on the PC
N = 30
kpc = []
for i in range(-N,N):
    for j in range(-N,N):
        kpc.append(i/N*b1 + j/N*b2)

kpc = np.array(kpc)

#differential
dkx = 2*np.pi/3/np.sqrt(len(kpc))/1000
dky = 2*np.pi/np.sqrt(3)/np.sqrt(len(kpc))/1000
my_dk = np.array([dkx, dky])
#NNNN
gap = 0.01 #[eV]
#calculate the derivatives of the condband on the grid
dpx = []
dpy = []
pp= np.load('c:/users/gugli/desktop/tesi/data/bande_bestfit.npy')
for k in kpc:
    null, dp = fun.ddEddP(k, pp, my_dk, gap)
    dpx.append(ddp[0,0,0,0,0].real)
    dpy.append(ddp[0,0,0,0,1].real)

dpx = np.array(dpx)
dpy = np.array(dpy)

my_dpi = 96
fig3, ax3= plt.subplots(figsize=(1.5*300/my_dpi, 1.5*300/my_dpi), dpi=my_dpi)
my_lw = 1.2
ax3.set_aspect('equal', 'box')

sc = plt.scatter(kpc[:,0],kpc[:,1], c = dpx, cmap = cmap, s = my_ms, marker = 'h')
plt.colorbar(sc, fraction = 0.045)
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
#plt.xticks(ticks, ticklabels)
#plt.yticks(ticks, ticklabels)
plt.title('$\partial^{2}_{xx}\, \mathrm{Re} P^{AA}_{+} (\mathbf{k})$ @$\Delta = %.3f$ eV'%(gap))
ax3.set_xlabel('$k_x a$')
ax3.set_ylabel('$k_y a$')
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.savefig('c:/users/gugli/desktop/tesi/figure/controlplot_ddpdxdx_square.png', dpi = my_dpi*5)

my_dpi = 96
fig4, ax4= plt.subplots(figsize=(1.5*300/my_dpi, 1.5*300/my_dpi), dpi=my_dpi)
my_lw = 1.2
ax4.set_aspect('equal', 'box')

cs = plt.scatter(kpc[:,0],kpc[:,1], c = dpy, cmap = cmap, s = my_ms, marker = 'h')
plt.colorbar(sc, fraction = 0.045)
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
ax4.set_xlabel('$k_x a$')
ax4.set_ylabel('$k_y a$')

plt.title('$\partial^{2}_{xy}\,  \mathrm{Re}  P^{AA}_{+} (\mathbf{k})$ @$\Delta = %.3f$ eV'%(gap))
#plt.xticks(ticks, ticklabels)
#plt.yticks(ticks, ticklabels)
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)

plt.savefig('c:/users/gugli/desktop/tesi/figure/controlplot_ddpdxdy_square.png', dpi = my_dpi*5)

plt.show()


##projector control plot on a square grid

cmap = plt.get_cmap('jet_r')
my_ms = 2

lim = 7
b1 = np.array([lim,0])
b2 = np.array([0,lim])

#build the grid on the PC
N = 100
kpc = []
for i in range(-N,N):
    for j in range(-N,N):
        kpc.append(i/N*b1 + j/N*b2)

kpc = np.array(kpc)

#NNNN
gap = 0.01 #[eV]
#calculate the derivatives of the condband on the grid
repv = []; impv = []; repc = []; impc = []
pp= np.load('c:/users/gugli/desktop/tesi/data/bande_bestfit.npy')
#pp= [0,pp[1],0,0,0,0,0]
for k in kpc:
    null, u, p1 = fun.EUP(k, pp, gap)
    repv1 = p1[0,0,1].real
    repv.append(repv1)
    impv1 = p1[0,0,1].imag
    impv.append(impv1)
    repc1 = p1[1,0,1].real
    repc.append(repc1)
    impc1 = p1[1,0,1].imag
    impc.append(impc1)

repv = np.array(repv)
impv = np.array(impv)
repc = np.array(repc)
impc = np.array(impc)

my_dpi = 96
fig3, ax3= plt.subplots(figsize=(1.5*300/my_dpi, 1.5*300/my_dpi), dpi=my_dpi)
my_lw = 1.2
ax3.set_aspect('equal', 'box')

sc = plt.scatter(kpc[:,0], kpc[:,1], c = repv, cmap = cmap, s = my_ms, marker = 'h')
plt.colorbar(sc, fraction = 0.045)
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
#plt.xticks(ticks, ticklabels)
#plt.yticks(ticks, ticklabels)
plt.title('$\mathrm{Re} P^{AB}_{-} (\mathbf{k})$ @$\Delta = %.3f$ eV'%(gap))
ax3.set_xlabel('$k_x a$')
ax3.set_ylabel('$k_y a$')
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.savefig('c:/users/gugli/desktop/tesi/figure/controlplot_repvAB.png', dpi = my_dpi*5)

my_dpi = 96
fig4, ax4 = plt.subplots(figsize=(1.5*300/my_dpi, 1.5*300/my_dpi), dpi=my_dpi)
ax4.set_aspect('equal', 'box')

sc = plt.scatter(kpc[:,0], kpc[:,1], c = repc, cmap = cmap, s = my_ms, marker = 'h')
plt.colorbar(sc, fraction = 0.045)
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
#plt.xticks(ticks, ticklabels)
#plt.yticks(ticks, ticklabels)
plt.title('$\mathrm{Re} P^{AB}_{+} (\mathbf{k})$ @$\Delta = %.3f$ eV'%(gap))
ax4.set_xlabel('$k_x a$')
ax4.set_ylabel('$k_y a$')
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.savefig('c:/users/gugli/desktop/tesi/figure/controlplot_repcaB.png', dpi = my_dpi*5)

my_dpi = 96
fig5, ax5= plt.subplots(figsize=(1.5*300/my_dpi, 1.5*300/my_dpi), dpi=my_dpi)
my_lw = 1.2
ax5.set_aspect('equal', 'box')

sc = plt.scatter(kpc[:,0], kpc[:,1], c = impv, cmap = cmap, s = my_ms, marker = 'h')
plt.colorbar(sc, fraction = 0.045)
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
#plt.xticks(ticks, ticklabels)
#plt.yticks(ticks, ticklabels)
plt.title('$\mathrm{Im} P^{AB}_{-} (\mathbf{k})$ @$\Delta = %.3f$ eV'%(gap))
ax5.set_xlabel('$k_x a$')
ax5.set_ylabel('$k_y a$')
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.savefig('c:/users/gugli/desktop/tesi/figure/controlplot_impvAB.png', dpi = my_dpi*5)

my_dpi = 96
fig6, ax6 = plt.subplots(figsize=(1.5*300/my_dpi, 1.5*300/my_dpi), dpi=my_dpi)
ax6.set_aspect('equal', 'box')

sc = plt.scatter(kpc[:,0], kpc[:,1], c = impc, cmap = cmap, s = my_ms, marker = 'h')
plt.colorbar(sc, fraction = 0.045)
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
#plt.xticks(ticks, ticklabels)
#plt.yticks(ticks, ticklabels)
plt.title('$\mathrm{Im} P^{AB}_{+} (\mathbf{k})$ @$\Delta = %.3f$ eV'%(gap))
ax6.set_xlabel('$k_x a$')
ax6.set_ylabel('$k_y a$')
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.savefig('c:/users/gugli/desktop/tesi/figure/controlplot_impcAB.png', dpi = my_dpi*5)


plt.show()

##NN analytic control plots

#NN vectors [in units of 1/a]
d0 = [[1/2, np.sqrt(3)/2],
    [1/2, -np.sqrt(3)/2],
    [-1,0]]

def sk(k):
    s = 0
    #NN geometric hopping factor
    for i in range(len(d0)):
        s += np.exp(-1j*k[0]*d0[i][0] - 1j*k[1]*d0[i][1])
    return s

cmap = plt.get_cmap('jet_r')
my_ms = 2

lim = 7
b1 = np.array([lim,0])
b2 = np.array([0,lim])

#build the grid on the PC
N = 100
kpc = []
for i in range(-N,N):
    for j in range(-N,N):
        kpc.append(i/N*b1 + j/N*b2)

kpc = np.array(kpc)

#differential
dkx = 2*np.pi/3/np.sqrt(len(kpc))/1000
dky = 2*np.pi/np.sqrt(3)/np.sqrt(len(kpc))/1000
my_dk = np.array([dkx, dky])
#NNNN
gap = 0.01 #[eV]
#calculate the derivatives of the condband on the grid
Th = []; reP = []; imP = []
for k in kpc:
    s = sk(k)
    th = np.angle(s)
    #qua c'è il segno - perché c'è un fattore -e^{-i*theta_k} negli autovettori numerici rispetto alla teoria
    repAB = -np.cos(th)/2
    impAB = np.sin(th)/2
    Th.append(th)
    reP.append(repAB)
    imP.append(impAB)

Th = np.array(Th)
reP = np.array(reP)
imP = np.array(imP)

my_dpi = 96
fig, ax= plt.subplots(figsize=(450/my_dpi, 450/my_dpi), dpi=my_dpi)
ax.set_aspect('equal', 'box')
#ax.axis('equal')

my_lw = 1.2

sc = plt.scatter(kpc[:,0], kpc[:,1], c = Th, cmap = cmap, s = my_ms, marker = 'h')
plt.colorbar(sc, fraction = 0.045)
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
#plt.xticks(ticks, ticklabels)
#plt.yticks(ticks, ticklabels)
plt.title(r'$\theta_{\mathbf{k}}$')
ax.set_xlabel('$k_x a$')
ax.set_ylabel('$k_y a$')
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.savefig('c:/users/gugli/desktop/tesi/figure/controlplot_thetak_analytic.png', dpi = my_dpi*5)

fig, ax= plt.subplots(figsize=(450/my_dpi, 450/my_dpi), dpi=my_dpi)
ax.set_aspect('equal', 'box')

my_lw = 1.2

sc = plt.scatter(kpc[:,0], kpc[:,1], c = reP, cmap = cmap, s = my_ms, marker = 'h')
plt.colorbar(sc, fraction = 0.045)
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
#plt.xticks(ticks, ticklabels)
#plt.yticks(ticks, ticklabels)
plt.title(r'Re $P_{+}^{AB}(\mathbf{k})$ @$\Delta = %.3f$ eV'%(gap))
ax.set_xlabel('$k_x a$')
ax.set_ylabel('$k_y a$')
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.savefig('c:/users/gugli/desktop/tesi/figure/controlplot_repcAB_analytic.png', dpi = my_dpi*5)

fig, ax= plt.subplots(figsize=(450/my_dpi, 450/my_dpi), dpi=my_dpi)
ax.set_aspect('equal', 'box')

my_lw = 1.2

sc = plt.scatter(kpc[:,0], kpc[:,1], c = imP, cmap = cmap, s = my_ms, marker = 'h')
plt.colorbar(sc, fraction = 0.045)
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
#plt.xticks(ticks, ticklabels)
#plt.yticks(ticks, ticklabels)
plt.title(r'Im $P_{+}^{AB}(\mathbf{k})$ @$\Delta = %.3f$ eV'%(gap))
ax.set_xlabel('$k_x a$')
ax.set_ylabel('$k_y a$')
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.savefig('c:/users/gugli/desktop/tesi/figure/controlplot_impcAB_analytic.png', dpi = my_dpi*5)

fig, ax= plt.subplots(figsize=(450/my_dpi, 450/my_dpi), dpi=my_dpi)
ax.set_aspect('equal', 'box')

my_lw = 1.2

sc = plt.scatter(kpc[:,0], kpc[:,1], c = reP, cmap = cmap, s = my_ms, marker = 'h')
plt.colorbar(sc, fraction = 0.045)
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
#plt.xticks(ticks, ticklabels)
#plt.yticks(ticks, ticklabels)
plt.title(r'Re $P_{+}^{AB}(\mathbf{k})$ @$\Delta = %.3f$ eV'%(gap))
ax.set_xlabel('$k_x a$')
ax.set_ylabel('$k_y a$')
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.savefig('c:/users/gugli/desktop/tesi/figure/controlplot_repcAB_analytic.png', dpi = my_dpi*5)

fig, ax= plt.subplots(figsize=(450/my_dpi, 450/my_dpi), dpi=my_dpi)
ax.set_aspect('equal', 'box')

my_lw = 1.2

sc = plt.scatter(kpc[:,0], kpc[:,1], c = imP, cmap = cmap, s = my_ms, marker = 'h')
plt.colorbar(sc, fraction = 0.045)
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
#plt.xticks(ticks, ticklabels)
#plt.yticks(ticks, ticklabels)
plt.title(r'Im $P_{+}^{AB}(\mathbf{k})$ @$\Delta = %.3f$ eV'%(gap))
ax.set_xlabel('$k_x a$')
ax.set_ylabel('$k_y a$')
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.savefig('c:/users/gugli/desktop/tesi/figure/controlplot_impcAB_analytic.png', dpi = my_dpi*5)

plt.show()

