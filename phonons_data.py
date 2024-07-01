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

data = np.loadtxt("c:/users/gugli/desktop/tesi/Per_Guglielmo_2D_cutoff/graphene_phonon_disp.gnuplot")

kk = np.unique(data[:, 0])
lines_data = np.reshape(data[:, 1], (-1, len(kk)))

#branch selection (0 = ZA-ZO, 1 = ZO-ZA)
print('Acquire data? \n')
acquire = input()
acquire_data = False
if acquire == 'y':
    acquire_data = True

which_branch = 5

k = []
e = []

def onclick(event):

    cx = event.xdata
    cy = event.ydata
    epsx = 0.005
    epsy = 10
    for i in range(len(lines_data)):
        for j in range(len(kk)):
            if kk[j] < cx + epsx and kk[j] > cx-epsx and lines_data[i,j] < cy + epsy and lines_data[i,j] > cy-epsy:
                if k:
                    already_in = False
                    for l in range(len(k)):
                        if kk[j] == k[l]:
                            already_in = True
                            print('Already in!')
                            break
                    if not already_in:
                        print(kk[j], lines_data[i,j])
                        print('index = %.0f\n'%(j))
                        k.append(kk[j])
                        e.append(lines_data[i,j])
                        break
                else:
                    print(kk[j], lines_data[i,j])
                    print('index = %.0f\n'%(j))
                    k.append(kk[j])
                    e.append(lines_data[i,j])
                    break

fig = plt.figure()

cmap = plt.get_cmap('jet')
for branch in range(len(lines_data)):
    plt.scatter(kk, lines_data[branch, :], c = lines_data[branch, :], s = 3)

#plt.xticks([0, 0.5774, 1.244, 1.5774], ['M','$\Gamma$','K', 'M'])
plt.grid(axis = 'x', linestyle = '--', alpha = 0.6, zorder = -2)
#plt.xlim(0,1.5774)

if acquire:
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

k = np.array(k)
e = np.array(e)

sort(k, e)

plt.plot(k, e, 'k.')

plt.show()

print('Save data?\n')
save = input()
save_data = False
if save == 'y':
    save_data = True

if save_data == True:
    if which_branch == 0:
        np.save(r'c:/users/gugli/desktop/tesi/data/ZA.npy', e)
    if which_branch == 1:
        np.save(r'c:/users/gugli/desktop/tesi/data/ZO.npy', e)
    if which_branch == 2:
        np.save(r'c:/users/gugli/desktop/tesi/data/TA.npy', e)
    if which_branch == 3:
        np.save(r'c:/users/gugli/desktop/tesi/data/LA.npy', e)
    if which_branch == 4:
        np.save(r'c:/users/gugli/desktop/tesi/data/TO.npy', e)
    if which_branch == 5:
        np.save(r'c:/users/gugli/desktop/tesi/data/LO.npy', e)





