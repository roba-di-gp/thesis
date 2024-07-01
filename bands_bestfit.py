import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import matplotlib
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
plt.rcParams['font.family'] = 'Palatino Linotype'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Palatino Linotype'
plt.rcParams['mathtext.it'] = 'Palatino Linotype:italic'
plt.rcParams['mathtext.bf'] = 'Palatino Linotype:bolditalic'
from scipy.optimize import minimize
import os
os.chdir('c:/users/gugli/desktop/tesi/codice')
import cmath
import functions as fun

#numbers
#c [nm/fs]
c = 299.792458
#hbar [eV fs]
hbar = 0.6582119569
#lattice parameter (from Guido) [nm]
lp = 0.2466731
#lattice step [nm]
a = lp/np.sqrt(3)
#lattice parameter/lattice step conversion factor/2pi
conv = 2*np.pi/np.sqrt(3)
#best guess for the parameters
#energy of the isolated 2p_z electron
epz = -4.2432
#hopping integrals (NN, NNN, NNNN)
tt = [-2.97, -0.073, -0.33]
#overlap integrals
ss = [0.073, 0.018, 0.026]

#import data
bands_data = [0,0]
k_data = np.load(r'c:/users/gugli/desktop/tesi/data/MGKM_sorted.npy')
bands_data[0] = np.load(r'c:/users/gugli/desktop/tesi/data/e1_sorted.npy')
bands_data[1] = np.load(r'c:/users/gugli/desktop/tesi/data/e0_sorted.npy')

#convert from MGKM to GKMG
k_data2 = []
bands_data2 = [[],[]]
for i in range(51, 166):
    k_data2.append(k_data[i]-0.5774)
for i in range(0, 51):
    k_data2.append(k_data[i]+1)
k_data = np.array(k_data2)
for i in range(2):
    for j in range(51, 166):
        bands_data2[i].append(bands_data[i][j])
    for j in range(0, 51):
        bands_data2[i].append(bands_data[i][j])
    bands_data2[i] = np.array(bands_data2[i])
bands_data = np.array(bands_data2)

#remove offset
bands_data = bands_data - bands_data[0,69]

#number of points in the contour
times = 1
lenGK = 70*times
lenKM = 46*times
lenMG = 52*times
#high symmetry contour
kxMG = np.linspace(2*np.pi/3, 0, lenMG)
kyMG = np.zeros(lenMG)
kxGK = np.linspace(0, 2*np.pi/3, lenGK)
kyGK = 1/np.sqrt(3)*kxGK
kyKM = np.linspace(2*np.pi/3/np.sqrt(3),0, lenKM)
kxKM = 2*np.pi/3*np.ones(lenKM)

kindGK = np.sqrt(kxGK**2+kyGK**2)/conv
kindKM = kindGK[-1] + np.flip(kyKM)/conv
kindMG = kindKM[-1] + np.flip(kxMG)/conv

kkx = np.concatenate((kxGK, kxKM[1:], kxMG[1:]))
kky = np.concatenate((kyGK, kyKM[1:], kyMG[1:]))

kind = np.concatenate((kindGK, kindKM[1:], kindMG[1:]))

for i in range(len(kind)):
    kind[i] = round(kind[i], 4)

kk = np.array([kkx, kky])
kk = np.transpose(kk)

#define and minimize a chi^2
def chi(pars):
    chi = 0
    for i in range(len(kind)):
        fk, null, null = fun.EUP(kk[i], pars)
        res0 = (bands_data[0][i]-fk[0])**2
        res1 = (bands_data[1][i]-fk[1])**2
        chi += res0+res1
    return chi

# popt = minimize(chi, [0, -2.97, -0.073, -0.33, 0.073, 0.018, 0.026], method='Nelder-Mead')
# print(popt.message)
# pp = popt.x
# np.save(r'c:/users/gugli/desktop/tesi/data/bande_bestfit.npy', pp)

pp = np.load('c:/users/gugli/desktop/tesi/data/bande_bestfit.npy')

#plot bands
my_dpi = 96
fig1, ax1 = plt.subplots(figsize=(1.8*350/my_dpi, 350/my_dpi), dpi=my_dpi)
my_lw = 1.5

my_ms = 4
dec = 2
plt.plot(k_data[::dec], bands_data[0][::dec], 'rh-', markersize = my_ms, zorder = 4, alpha = 0.5, linewidth=my_lw, label ='DFT')
plt.plot(k_data[::dec], bands_data[1][::dec], 'rh-', markersize = my_ms, zorder = 4, alpha = 0.5, linewidth=my_lw)

#NNNN
#calculate the dispersion on the high-symmetry contour
E = []
P = []
dP = []
dk = np.array([1,1])*(kk[1]-kk[0])/50
for i in range(len(kk)):
    e, u, p = fun.EUP(kk[i], pp)
    de, dp = fun.dEdP(kk[i], pp, dk)
    E.append(e)
    P.append(p)
    dP.append(dp)

E = np.array(E)

plt.plot(kind, E[:,0], 'k-', linewidth=my_lw, zorder=4, label = 'Best fit')
plt.plot(kind, E[:,1], 'k-', linewidth=my_lw, zorder=4)

#figure details
ticks = [0, 0.6666, 1., 1.5774]
ticklabels = ['$\Gamma$','K', 'M', '$\Gamma$']
plt.xticks(ticks, ticklabels)
ax1.set_ylabel('Energy  [eV]')

plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.1, top = 0.9)
#plt.legend(shadow=True, loc = 'upper center', prop={'size': 8})
plt.grid(axis = 'x', linestyle = '--', alpha = 0.6, zorder = -2)
plt.xlim(0, 1.5774)
plt.legend(shadow=True, loc = (0.4,0.7), prop={'size': 12})

plt.savefig('c:/users/gugli/desktop/tesi/figure/pi_bands_DFT.jpeg', dpi = my_dpi*5)

plt.show()

##Dirac point plot

#redo the contour around K
ext = 0.01
lenk = 1000
ky = np.linspace(4*np.pi/3/np.sqrt(3)-ext, 4*np.pi/3/np.sqrt(3)+ext, lenk)
kx = np.zeros(lenk)
kk = np.array([kx, ky])
kk = np.transpose(kk)

kind = np.linspace(0, 1, lenk)
my_lw = 1.5

my_ms = 3.
dec = 3

#NNNN
#calculate the dispersion on the high-symmetry contour
gap = 0.02
E = []
EG = []
dk = np.array([1,1])*(kk[1]-kk[0])/50
pp = np.load('c:/users/gugli/desktop/tesi/data/bande_bestfit.npy')
#pp = [0,pp[1],0,0,0,0,0]
for i in range(len(kk)):
    e, u, p = fun.EUP(kk[i], pp)
    eg, u, p = fun.EUP(kk[i], pp, gap)
    E.append(e)
    EG.append(eg)

EG = np.array(EG) - E[int(lenk/2)-1][0]
E = np.array(E) - E[int(lenk/2)-1][0]

#Fermi velocity [in units of a]
v = -3/2*pp[1]/hbar
#mass
m = gap/2/v**2
#compton wavelength [in units of a]
lambdabar = -3*pp[1]/gap

cono = hbar*v*abs(ky-4*np.pi/3/np.sqrt(3))
parabola = np.sqrt(m**2*v**4 + hbar**2*v**2*(ky-4*np.pi/3/np.sqrt(3))**2)
nonrel = m*v**2 + hbar**2*(ky-4*np.pi/3/np.sqrt(3))**2/2/m

#plot bands
my_dpi = 96
fig1, ax1 = plt.subplots(figsize=(500/my_dpi, 310/my_dpi), dpi=my_dpi)

#NNNN bands (ungapped)
plt.plot(kind, E[:,0], 'k-', linewidth=my_lw, zorder=4)
plt.plot(kind, E[:,1], 'k-', linewidth=my_lw, zorder=4, label = 'NNNN, $\Delta = 0$')

#NNNN bands (gapped)
plt.plot(kind, EG[:,0], 'k-', linewidth=my_lw*1.5, zorder=3)
plt.plot(kind, EG[:,1], 'k-', linewidth=my_lw*1.5, zorder=3)
plt.plot(kind, EG[:,0], 'r-', linewidth=my_lw, zorder=4)
plt.plot(kind, EG[:,1], 'r-', linewidth=my_lw, zorder=4, label = 'NNNN, $\Delta = 20$ meV')

#NN gapped non relativistic
plt.plot(kind, nonrel, 'r:', linewidth=my_lw, zorder=1)
plt.plot(kind, -nonrel, 'r:', linewidth=my_lw, zorder=1, label = 'NN non-rel, $\Delta = 20$ meV')

#figure details
ticks = [0., 0.4975, 1.0]
ticklabels = ['$Ka-0.005$','$Ka$','$Ka + 0.005$',]
plt.xticks(ticks, ticklabels)
ax1.set_ylabel('Energy  [eV]')
plt.plot(kind, np.zeros(lenk), 'k--', alpha=0.6, linewidth=0.7)

#plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.15, top = 0.85)
plt.grid(axis = 'x', linestyle = '--', alpha = 0.6, zorder = -2)
plt.xlim(0, 1)
plt.ylim(-0.04, 0.04)
plt.legend(shadow=True, loc = (0.45,0.85), prop={'size': 10})

plt.savefig('c:/users/gugli/desktop/tesi/figure/pi_bands_diracpoint.jpeg', dpi = my_dpi*5)


plt.show()

##3D surface plot
from matplotlib import cbook, cm
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

my_dpi = 96
fig = plt.figure(figsize=(1.5*300*(1+np.sqrt(5))/2/my_dpi, 1.5*300/my_dpi), dpi=my_dpi)
ax = plt.axes(projection = '3d')
my_lw = 1.2

vv = [
    [0, 4*np.pi/3/np.sqrt(3)],
    [2*np.pi/3, 2*np.pi/3/np.sqrt(3)],
    [2*np.pi/3, -2*np.pi/3/np.sqrt(3)],
    [0, -4*np.pi/3/np.sqrt(3)],
    [-2*np.pi/3, -2*np.pi/3/np.sqrt(3)],
    [-2*np.pi/3, 2*np.pi/3/np.sqrt(3)]
]

lim = 3
N = 400
kx = np.linspace(-lim, lim, N)
ky = np.linspace(-lim, lim, N)

kkx, kky = np.meshgrid(kx, ky)

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

#NNNN
#calculate the dispersion on the FBZ grid
E0 = np.zeros((len(kx),len(kx)))
E1 = np.zeros((len(kx),len(kx)))
for i in range(len(kx)):
    for j in range(len(kx)):
        E0[i,j] = -2.97*abs(sk([kx[i],ky[j]]))
        E1[i,j] = 2.97*abs(sk([kx[i],ky[j]]))

cmap0 = cm.bone_r
cmap1 = cm.bone
ls = LightSource(40, -250)
# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
rgb = ls.shade(E0, cmap=cmap0, vert_exag=0.1, vmin = -11, vmax = 0,  blend_mode='soft')
ax.plot_surface(kkx,kky, E0, rstride=1, cstride=1, facecolors=rgb,linewidth=0, antialiased=False, shade=False)
rgb = ls.shade(E1, cmap=cmap1, vert_exag=0.1, vmin = 0, vmax = 11,  blend_mode='soft')
ax.plot_surface(kkx,kky, E1, rstride=1, cstride=1, facecolors=rgb,linewidth=0, antialiased=False, shade=False)

# for i in range(len(vv)-1):
#     k1x = np.linspace(vv[i][0], vv[i+1][0], 2)
#     k1y = np.linspace(vv[i][1], vv[i+1][1], 2)
#     ax.plot3D(k1x, k1y, 0, 'k-')

plt.xlim(-lim,lim)
plt.ylim(-lim,lim)
#figure details
ax.view_init(8, -250)
ax.set_xlabel('$k_y a$')
ax.set_ylabel('$k_x a$')
ax.set_zlabel('Energy  [eV]', rotation = 180)
plt.subplots_adjust(left=0., right=1., bottom = 0., top = 1.)
ax.grid(True)
ax.xaxis._axinfo["grid"].update({"linewidth":0.3})
ax.yaxis._axinfo["grid"].update({"linewidth":0.3})
ax.zaxis._axinfo["grid"].update({"linewidth":0.3})
#plt.savefig('c:/users/gugli/desktop/tesi/figure/pi_bands_surface.png', dpi = my_dpi*5)

plt.show()

##colorplots

cmap = plt.get_cmap('jet_r')
my_ms = 3

K = 4*np.pi/3/np.sqrt(3)

lim = 3.2
b1 = np.array([lim,0])
b2 = np.array([0,lim])

#build the grid on the PC
N = 70
kpc = []
for i in range(-N,N):
    for j in range(-N,N):
        kpc.append(i/N*b1 + j/N*b2)

kpc = np.array(kpc)

#NNNN
gap = 0. #[eV]
#calculate the energy
ec = []
pp= np.load('c:/users/gugli/desktop/tesi/data/bande_bestfit.npy')
#pp= [0,pp[1],0,0,0,0,0]
for k in kpc:
    e, u, p = fun.EUP(k, pp, gap)
    ec.append(e[0])

ec = np.array(ec)

my_dpi = 96
fig, ax= plt.subplots(figsize=(1.5*300/my_dpi, 1.5*300/my_dpi), dpi=my_dpi)
my_lw = 1.2
ax.set_aspect('equal', 'box')

sc = plt.scatter(kpc[:,0], kpc[:,1], c = ec, cmap = cmap, s = my_ms, marker = 'h')
plt.colorbar(sc, fraction = 0.045)
plt.subplots_adjust(left=0.2, right=0.9, bottom = 0.2, top = 0.9)
if gap > 0:
    plt.title('$E_{-} (\mathbf{k})$ @$\Delta = %.0f$ meV'%(gap*1000))
if gap == 0:
    plt.title('$E_{-} (\mathbf{k})$ @$\Delta = 0$')

ax.set_xlabel('$k_x a$')
ax.set_ylabel('$k_y a$')
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)


# #rebuild the grid for the zoom
# limz = 0.1
# b1 = np.array([limz,0])
# b2 = np.array([0,limz])
#
# N = 70
# kpcz = []
# for i in range(-N,N):
#     for j in range(-N,N):
#         kpcz.append(i/N*b1 + j/N*b2)
#
# kpcz = np.array(kpcz)
# kpcz = kpcz + np.array([0,K])
#
# #NNNN
# #calculate the condband projector
# ecz = []
# #pp= [0,pp[1],0,0,0,0,0]
# for k in kpcz:
#     e, u, p = fun.EUP(k, pp, gap)
#     ecz.append(e[0])
#
# ecz = np.array(ecz)
#
# axins = ax.inset_axes([-3.2*lim, -0.7*lim, 1.4*lim, 1.4*lim], transform=ax.transData)
# axins.scatter(kpcz[:,0], kpcz[:,1], c = ecz, cmap = cmap, s = my_ms, marker = 'h')
# axins.set_xlim(-limz, limz)
# axins.set_ylim(K-limz, K+limz)
# # axins.set_xticks([])
# # axins.set_yticks([])
# # axins.set_xticklabels([])
# # axins.set_yticklabels([])
# ax.indicate_inset_zoom(axins, edgecolor='k', alpha=0.7)
#

gapstr = str(gap*1000)
path = 'c:/users/gugli/desktop/tesi/figure/valband' + gapstr + 'meV.png'
plt.savefig(path, dpi = my_dpi*5)

plt.show()

##bands control plot on the FBZ grid

cmap = plt.get_cmap('jet')
lim = 2.6
my_ms = 32
N = 30

fbzmesh = fun.fbz_meshgrid(N,1,1)
#NNNN
gap = 0.01 #[eV]
#calculate the derivatives of the condband on the grid
ec = []; ev = []
pp= np.load('c:/users/gugli/desktop/tesi/data/bande_bestfit.npy')
for k in fbzmesh:
    e, null, null = fun.EUP(k, pp, gap)
    ev.append(e[0])
    ec.append(e[1])

ev = np.array(ev)
ec = np.array(ec)

my_dpi = 96
fig3, ax3= plt.subplots(figsize=(1.5*300/my_dpi, 1.5*300/my_dpi), dpi=my_dpi)
ax3.set_aspect('equal', 'box')
my_lw = 1.2

sc = plt.scatter(fbzmesh[:,0],fbzmesh[:,1], c = ec, cmap = cmap, s = my_ms, marker = 'h')
plt.colorbar(sc, fraction = 0.045, label='Energy [eV]')
plt.text(1.4,2.1, r'%.0f$\times$%.0f'%(N,N))
plt.subplots_adjust(left=0.15, right=0.85, bottom = 0.15, top = 0.85)
ax3.set_xlabel('$k_x a$')
ax3.set_ylabel('$k_y a$')

plt.title(r'$E_{+} (\mathbf{k})$, $\Delta = %.0f$ meV'%(gap*1e3))
#plt.xticks(ticks, ticklabels)
#plt.yticks(ticks, ticklabels)
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)

plt.savefig('c:/users/gugli/desktop/tesi/figure/controlplot_ec_fbz.png', dpi = my_dpi*5)

my_dpi = 96
fig4, ax4= plt.subplots(figsize=(1.5*300/my_dpi, 1.5*300/my_dpi), dpi=my_dpi)
ax4.set_aspect('equal', 'box')
my_lw = 1.2

sc = plt.scatter(fbzmesh[:,0],fbzmesh[:,1], c = ev, cmap = cmap, s = my_ms, marker = 'h')
plt.colorbar(sc, fraction = 0.045, label='Energy [eV]')
plt.text(1.4,2.1, r'%.0f$\times$%.0f'%(N,N))
plt.subplots_adjust(left=0.15, right=0.85, bottom = 0.15, top = 0.85)
ax4.set_xlabel('$k_x a$')
ax4.set_ylabel('$k_y a$')
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)

plt.title('$ E_{-} (\mathbf{k})$, $\Delta = %.0f$ meV'%(gap*1e3))
#plt.xticks(ticks, ticklabels)
#plt.yticks(ticks, ticklabels)

plt.savefig('c:/users/gugli/desktop/tesi/figure/controlplot_ev_fbz.png', dpi = my_dpi*5)

plt.show()


##derivatives control plot on the FBZ grid

cmap = plt.get_cmap('jet')
lim = 2.6
my_ms = 20

fbzmesh = fun.fbz_meshgrid(28,1,1)
#differential
dkx = 2*np.pi/3/np.sqrt(len(fbzmesh))/1000
dky = 2*np.pi/np.sqrt(3)/np.sqrt(len(fbzmesh))/1000
my_dk = np.array([dkx, dky])
#NNNN
gap = 0.01 #[eV]
#calculate the derivatives of the condband on the grid
dex = []
dey = []
pp= np.load('c:/users/gugli/desktop/tesi/data/bande_bestfit.npy')
for k in fbzmesh:
    de1, null  = fun.dEdP(k, pp,my_dk, gap)
    dex.append(de1[0,1].real)
    dey.append(de1[1,1].real)

dex = np.array(dex)
dey = np.array(dey)

my_dpi = 96
fig3, ax3= plt.subplots(figsize=(1.5*300/my_dpi, 1.5*300/my_dpi), dpi=my_dpi)
ax3.set_aspect('equal', 'box')
my_lw = 1.2

sc = plt.scatter(fbzmesh[:,0],fbzmesh[:,1], c = dex, cmap = cmap, s = my_ms, marker = 'h')
plt.colorbar(sc, fraction = 0.045)
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
ax3.set_xlabel('$k_x a$')
ax3.set_ylabel('$k_y a$')

plt.title('$ \partial_{x} E_{+} (\mathbf{k})$, $\Delta = %.3f$ eV'%(gap))
#plt.xticks(ticks, ticklabels)
#plt.yticks(ticks, ticklabels)
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)

plt.savefig('c:/users/gugli/desktop/tesi/figure/controlplot_dedx_fbz.png', dpi = my_dpi*5)

my_dpi = 96
fig4, ax4= plt.subplots(figsize=(1.5*300/my_dpi, 1.5*300/my_dpi), dpi=my_dpi)
ax4.set_aspect('equal', 'box')
my_lw = 1.2

sc = plt.scatter(fbzmesh[:,0],fbzmesh[:,1], c = dey, cmap = cmap, s = my_ms, marker = 'h')
plt.colorbar(sc, fraction = 0.045)
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.2, top = 0.8)
ax4.set_xlabel('$k_x a$')
ax4.set_ylabel('$k_y a$')
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)

plt.title('$ \partial_{y} E_{+} (\mathbf{k})$, $\Delta = %.3f$ eV'%(gap))
#plt.xticks(ticks, ticklabels)
#plt.yticks(ticks, ticklabels)

plt.savefig('c:/users/gugli/desktop/tesi/figure/controlplot_dedy_fbz.png', dpi = my_dpi*5)

plt.show()

##DFT bands plot

data = np.loadtxt("c:/users/gugli/desktop/tesi/Per_Guglielmo_2D_cutoff/Bande_graphene.gnu")

k_data = np.unique(data[:, 0])
bands_data = np.reshape(data[:, 1], (-1, len(kk)))
my_dpi = 96
fig, ax= plt.subplots(figsize=(1.8*350/my_dpi, 350/my_dpi), dpi=my_dpi)

#convert from MGKM to GKMG
k_data2 = []
bands_data2 = []
for i in range(6):
    bands_data2.append([])
for i in range(51, 166):
    k_data2.append(k_data[i]-0.5774)
for i in range(0, 51):
    k_data2.append(k_data[i]+1)
k_data = np.array(k_data2)
for i in range(len(bands_data2)):
    for j in range(51, 166):
        bands_data2[i].append(bands_data[i][j])
    for j in range(0, 51):
        bands_data2[i].append(bands_data[i][j])
    bands_data2[i] = np.array(bands_data2[i])
bands_data = np.array(bands_data2)

lw = 1.2
for i in range(len(bands_data)):
    plt.plot(kind, bands_data[i]- bands_data[3][69],'k-', linewidth = lw)

plt.plot(kind[64:77], bands_data[3][64:77]- bands_data[3][69],'r-', linewidth = lw)
plt.plot(kind[64:77], bands_data[4][64:77]- bands_data[3][69],'r-', linewidth = lw)

#figure details
ticks = [0, 0.6666, 1., 1.5774]
ticklabels = ['$\Gamma$','K', 'M', '$\Gamma$']
plt.xticks(ticks, ticklabels)
ax.set_ylabel('Energy  [eV]')

plt.plot(kind, np.zeros(len(kind)), color ='gray',linestyle='--', alpha=0.6, linewidth=0.7, zorder = -3)
plt.subplots_adjust(left=0.2, right=0.8, bottom = 0.1, top = 0.9)
#plt.legend(shadow=True, loc = 'upper center', prop={'size': 8})
plt.grid(axis = 'x', linestyle = '--', alpha = 0.6, zorder = -2)
plt.xlim(0, 1.5774)
plt.ylim(-20, 5.8)
plt.savefig('c:/users/gugli/desktop/tesi/figure/graphene_bands_Guido.png', dpi = my_dpi*5)

plt.show()
