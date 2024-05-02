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
import os
os.chdir('c:/users/gugli/desktop/tesi/codice')
from functions import *

def sort(arr, arr2):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                arr2[j], arr2[j+1] = arr2[j+1], arr2[j]

data = np.loadtxt("c:/users/gugli/desktop/tesi/Per_Guglielmo_2D_cutoff/Bande_graphene.gnu")

kk = np.unique(data[:, 0])
bands = np.reshape(data[:, 1], (-1, len(kk)))

#band selection (0 = conduction band, 1 = valence band)
#print('Save data? \n')
#save = input()

save_data = False
if save == 'y':
    save_data = True
which_band = 1

k = []
e = []

def onclick(event):

    cx = event.xdata
    cy = event.ydata
    epsx = 0.002
    epsy = 0.1
    for i in range(len(bands)):
        for j in range(len(kk)):
            if kk[j] < cx + epsx and kk[j] > cx-epsx and bands[i,j] < cy + epsy and bands[i,j] > cy-epsy:
                if k:
                    already_in = False
                    for l in range(len(k)):
                        if kk[j] == k[l]:
                            already_in = True
                            print('Already in!')
                            break
                    if not already_in:
                        print(kk[j], bands[i,j])
                        print('index = %.0f\n'%(j))
                        k.append(kk[j])
                        e.append(bands[i,j])
                        break
                else:
                    print(kk[j], bands[i,j])
                    print('index = %.0f\n'%(j))
                    k.append(kk[j])
                    e.append(bands[i,j])
                    break

fig = plt.figure()

for band in range(len(bands)):
    plt.plot(kk, bands[band, :], 'k.', linewidth=1, alpha=1, markersize = 3)

plt.xticks([0, 0.5774, 1.244, 1.5774], ['M','$\Gamma$','K', 'M'])
plt.grid(axis = 'x', linestyle = '--', alpha = 0.6, zorder = -2)
plt.xlim(0,1.5774)
#cid = fig.canvas.mpl_connect('button_press_event', onclick)

#plt.show()

k = np.array(k)
e = np.array(e)

if save_data == True:
    if which_band == 0:
        np.save(r'c:/users/gugli/desktop/tesi/data/e0.npy', e)
        np.save(r'c:/users/gugli/desktop/tesi/data/MGKM0.npy', k)
    if which_band == 1:
        np.save(r'c:/users/gugli/desktop/tesi/data/e1.npy', e)
        np.save(r'c:/users/gugli/desktop/tesi/data/MGKM1.npy', k)


k0 = np.load(r'c:/users/gugli/desktop/tesi/data/MGKM0.npy')
k1 = np.load(r'c:/users/gugli/desktop/tesi/data/MGKM1.npy')
e0 = np.load(r'c:/users/gugli/desktop/tesi/data/e0.npy')
e1 = np.load(r'c:/users/gugli/desktop/tesi/data/e1.npy')

sort(k0, e0)
sort(k1, e1)

np.save(r'c:/users/gugli/desktop/tesi/data/MGKM_sorted.npy', k0)
np.save(r'c:/users/gugli/desktop/tesi/data/e0_sorted.npy', e0)
np.save(r'c:/users/gugli/desktop/tesi/data/e1_sorted.npy', e1)

plt.plot(k0, e0, 'r.')
plt.plot(k1, e1, 'r.')

plt.show()



