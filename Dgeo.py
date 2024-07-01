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
import os
os.chdir('c:/users/gugli/desktop/tesi/codice')
from functions import *
from time import time
timezero = time()

##values in eV

print('Defining/importing numerical values \n')
#hbar [eV s]
hbar = 6.582119569e-16
#lattice parameter [hbar*c/eV]
lp = 2.466731e-10/1.97327e-7
#lattice step [hbar*c/eV]
a = lp/np.sqrt(3)
#lattice parameter/lattice step conversion factor * 2pi
conv_lattice = 2*np.pi/np.sqrt(3)
#carbon atom mass [eV/c^2]
M = 12*1.66054e-27/1.78266192e-36

#best fit parameters for electron bands [eV]
pp_el = np.load('c:/users/gugli/desktop/tesi/data/bande_bestfit.npy')

#best fit parameters for phonon lines [eV^2]
pp_ph = np.load(r'c:/users/gugli/desktop/tesi/data/phonons_bestfit.npy')

#gamma parameters [hbar^2 c^2 / eV^2]
gamma = np.load(r'c:/users/gugli/desktop/tesi/data/gamma.npy')
gamma_mat = np.array([np.array([gamma[0], gamma[1]]), np.array([gamma[1], gamma[0]])])
gamma_mat = gamma_mat*1e20*(1.97327e-7)**2

#FBZ meshgrid [1/a]
fbz = np.load(r'c:/users/gugli/desktop/tesi/data/fbz_meshgrid.npy')

#differential [1/a]
dkx = 2*np.pi/3/np.sqrt(len(fbz))/10
dky = 2*np.pi/np.sqrt(3)/np.sqrt(len(fbz))/10
dk = [dkx, dky]

#temperature [eV]
T = hbar*10**6*dky*10**10/10
#high-symmetry contour
#number of points in the contour
times = 1
lenMG = 30*times
lenGK = 41*times
lenKM = 20*times
#high symmetry contour
qxMG = np.linspace(2*np.pi/3, 0, lenMG)
qyMG = np.zeros(lenMG)
qxGK = np.linspace(0, 2*np.pi/3, lenGK)
qyGK = 1/np.sqrt(3)*qxGK
qyKM = np.linspace(2*np.pi/3/np.sqrt(3),0, lenKM)
qxKM = 2*np.pi/3*np.ones(lenKM)

qindGK = np.sqrt(qxGK**2+qyGK**2)/conv_lattice
qindKM = qindGK[-1] + np.flip(qyKM)/conv_lattice
qindMG = qindKM[-1] + np.flip(qxMG)/conv_lattice

qqx = np.concatenate((qxGK, qxKM, qxMG))
qqy = np.concatenate((qyGK, qyKM, qyMG))

qind = np.concatenate((qindGK, qindKM, qindMG))

for i in range(len(qind)):
    qind[i] = round(qind[i], 4)

qq = np.array([qqx, qqy])
qq = np.transpose(qq)


##evaluation

print('Save data? \n')
save = input()
save_data = False
if save == 'y':
    save_data = True

start = time()
#initialize variables with the right dimensions
#E(k), u(k), P(k)
ek = np.zeros((len(fbz), 2)); uk = np.zeros((len(fbz), 2,2), dtype = 'complex'); pk = np.zeros((len(fbz), 2, 2, 2), dtype = 'complex')
#E(-k), u(-k), P(-k)
emk = np.zeros((len(fbz), 2)); umk = np.zeros((len(fbz), 2,2), dtype = 'complex'); pmk = np.zeros((len(fbz), 2, 2, 2), dtype = 'complex')
#E(k-q), u(k-q), P(k-q)
ekmq = np.zeros((len(qq), len(fbz), 2)); ukmq = np.zeros((len(qq), len(fbz),2,2), dtype = 'complex'); pkmq = np.zeros((len(qq), len(fbz), 2, 2, 2), dtype = 'complex')
#E(q-k), u(q-k), P(q-k)
eqmk = np.zeros((len(qq), len(fbz), 2)); uqmk = np.zeros((len(qq), len(fbz),2,2), dtype = 'complex'); pqmk = np.zeros((len(qq), len(fbz), 2, 2, 2), dtype = 'complex')
#dE(k), dP(k)
dek = np.zeros((len(fbz), 2, 2), dtype = 'complex'); dpk = np.zeros((len(fbz), 2, 2, 2, 2), dtype = 'complex')
#dE(-k), dP(-k)
demk = np.zeros((len(fbz), 2, 2), dtype = 'complex'); dpmk = np.zeros((len(fbz), 2, 2, 2, 2), dtype = 'complex')
#dE(k-q), dP(k-q)
dekmq = np.zeros((len(qq), len(fbz), 2, 2), dtype = 'complex'); dpkmq = np.zeros((len(qq), len(fbz), 2, 2, 2, 2), dtype = 'complex')
#dE(q-k), dP(q-k)
deqmk = np.zeros((len(qq), len(fbz), 2, 2), dtype = 'complex'); dpqmk = np.zeros((len(qq), len(fbz), 2, 2, 2, 2), dtype = 'complex')
#ddE(k), ddP(k)
ddek = np.zeros((len(fbz), 2, 2, 2), dtype = 'complex'); ddpk = np.zeros((len(fbz), 2, 2, 2, 2, 2), dtype = 'complex')
#ddE(q-k), ddP(q-k)
ddeqmk = np.zeros((len(qq), len(fbz), 2, 2, 2), dtype = 'complex'); ddpqmk = np.zeros((len(qq), len(fbz), 2, 2, 2, 2, 2), dtype = 'complex')

print('Evaluating and vectorizing relevant quantities \n')
#cycle on the FBZ
for k_ind in range(len(fbz)):
    k = fbz[k_ind,:]
    ek[k_ind], uk[k_ind], pk[k_ind] = EUP(k, pp_el)
    emk[k_ind], umk[k_ind], pmk[k_ind] = EUP(-k, pp_el)
    dek[k_ind], dpk[k_ind] = dEdP(k, pp_el, dk)
    demk[k_ind], dpmk[k_ind] = dEdP(-k, pp_el, dk)
    ddek[k_ind], ddpk[k_ind] = dEdP(k, pp_el, dk)
    #cycle on the high-symmetry contour
    for q_ind in range(len(qq)):
        q = qq[q_ind,:]
        ekmq[q_ind, k_ind], ukmq[q_ind, k_ind], pkmq[q_ind, k_ind] = EUP(k-q, pp_el)
        eqmk[q_ind, k_ind], uqmk[q_ind, k_ind], pqmk[q_ind, k_ind] = EUP(q-k, pp_el)
        dekmq[q_ind, k_ind], dpkmq[q_ind, k_ind] = dEdP(k-q, pp_el, dk)
        deqmk[q_ind, k_ind], dpqmk[q_ind, k_ind] = dEdP(q-k, pp_el, dk)
        ddeqmk[q_ind, k_ind], ddpqmk[q_ind, k_ind] = ddEddP(q-k, pp_el, dk)
    print('%.2f'%(k_ind/len(fbz)))

print('Time for evaluation: %.1f min'%((time()-start)/60))

##calculation of the matrix elements

print('Calculating F/M tensors\n')
#F[q index, k index, band index 1, band index 2, sublattice index, cartesian index]
#FE1 = FE^{nu k, nu' k-q}_{tau i}
#FE2 = FE^{nu k-q, nu' k}_{tau' j}
#FG1 = FG^{nu k, nu' k-q}_{tau i}
#FG2 = FG^{nu k-q, nu' k}_{tau' j}
#M1[k index, sublattice index 1, sublattice index 2, cartesian index 1, cartesian index 2]
#M2[q index, k index, sublattice index 1, sublattice index 2, cartesian index 1, cartesian index 2]
#MG1 = MG(k)_{ij}_{tau tau'}
#MEG1 = MEG(k)_{ij}_{tau tau'}
#MG2 = MG(q-k)_{ij}_{tau tau'}
#MEG2 = MEG(q-k)_{ij}_{tau tau'}
FE1 = np.zeros((len(qq), len(fbz), 2, 2, 2, 2), dtype = 'complex')
FE2 = np.zeros((len(qq), len(fbz), 2, 2, 2, 2), dtype = 'complex')
FG1 = np.zeros((len(qq), len(fbz), 2, 2, 2, 2), dtype = 'complex')
FG2 = np.zeros((len(qq), len(fbz), 2, 2, 2, 2), dtype = 'complex')
MG1 = np.zeros((len(fbz), 2, 2, 2, 2), dtype = 'complex')
MEG1 = np.zeros((len(fbz), 2, 2, 2, 2), dtype = 'complex')
MG2 = np.zeros((len(qq), len(fbz), 2, 2, 2, 2), dtype = 'complex')
MEG2 = np.zeros((len(qq), len(fbz), 2, 2, 2, 2), dtype = 'complex')
start = time()
#cycle on the FBZ
for k_ind in range(len(fbz)):
    #cycle on the high-symmetry contour
    for q_ind in range(len(qq)):
        #cycle on the F tensor indices
        for nu1 in range(2):
            for nu2 in range(2):
                for tau in range(2):
                    for i in range(2):
                        #sum on sublattice and band
                        for nux in range(2):
                            for taux in range(2):
                                FE1[q_ind, k_ind, nu1, nu2, tau, i] += 1j*gamma_mat[tau, taux]*(ukmq[q_ind, k_ind, nu2, tau]*dek[k_ind, i, nux]*pk[k_ind, nux, tau, taux]*np.conj(uk[k_ind, nu1, taux]) + 1j*np.conj(uk[k_ind, nu1, tau])*deqmk[q_ind, k_ind, i, nux]*(pqmk[q_ind, k_ind, nux, tau, taux])*(uqmk[q_ind, k_ind, nu2, taux]))
                                FE2[q_ind, k_ind, nu1, nu2, tau, i] += 1j*gamma_mat[tau, taux]*(uk[k_ind, nu2, tau]*dekmq[q_ind, k_ind, i, nux]*pkmq[q_ind, k_ind, nux, tau, taux]*np.conj(ukmq[q_ind, k_ind, nu1, taux]) + 1j*np.conj(ukmq[q_ind, k_ind, nu1, tau])*demk[k_ind, i, nux]*pmk[k_ind, nux, tau, taux]*(umk[k_ind, nu2, taux]))
                                FG1[q_ind, k_ind, nu1, nu2, tau, i] += 1j*gamma_mat[tau, taux]*(ukmq[q_ind, k_ind, nu2, tau]*ek[k_ind, nux]*dpk[k_ind, i, nux, tau, taux]*np.conj(uk[k_ind, nu1, taux]) + 1j*np.conj(uk[k_ind, nu1, tau])*eqmk[q_ind, k_ind, nux]*(dpqmk[q_ind,k_ind, i, nux, tau, taux])*(uqmk[q_ind,k_ind, nu2, taux]))
                                FG2[q_ind, k_ind, nu1, nu2, tau, i] += 1j*gamma_mat[tau, taux]*(uk[k_ind, nu2, tau]*ekmq[q_ind, k_ind, nux]*dpkmq[q_ind, k_ind, i, nux, tau, taux]*np.conj(ukmq[q_ind, k_ind, nu1, taux]) + 1j*np.conj(ukmq[q_ind, k_ind, nu1, tau])*emk[k_ind, nux]*(dpmk[k_ind,i, nux, tau, taux])*(umk[k_ind, nu2, taux]))
    #cycle on the M tensor indices
    for tau1 in range(2):
        for tau2 in range(2):
            for i in range(2):
                for j in range(2):
                    #sum on bands
                    for nux in range(2):
                        MG1[k_ind, tau1, tau2, i, j] += - gamma_mat[tau1, tau2]**2*ek[k_ind, nux]*ddpk[k_ind, i,j,nux, tau1, tau2]
                        MEG1[k_ind, tau1, tau2, i, j] += - gamma_mat[tau1, tau2]**2*dek[k_ind, i, nux]*dpk[k_ind,j,nux, tau1, tau2]
                    for q_ind in range(len(qq)):
                        for nux in range(2):
                            MG2[q_ind, k_ind, tau1, tau2, i, j] += - gamma_mat[tau1, tau2]**2*eqmk[q_ind,k_ind, nux]*ddpqmk[q_ind,k_ind, i,j,nux, tau1, tau2]
                            MEG2[q_ind, k_ind, tau1, tau2, i, j] += - gamma_mat[tau1, tau2]**2*deqmk[q_ind, k_ind, i, nux]*dpqmk[q_ind, k_ind,j,nux, tau1, tau2]
    print('%.2f'%(k_ind/len(fbz)))

print('Calculating matrix elements\n')
Dgeo = []
#cycle on the high-symmetry contour
for q_ind in range(len(qq)):
    DG = np.zeros((2,2,2,2), dtype = 'complex')
    #cycle on the dynamical matrix indices
    for tau1 in range(2):
        for tau2 in range(2):
            for i in range(2):
                for j in range(2):
                    #sum on the FBZ
                    for k_ind in range(len(fbz)):
                        DG[tau1, tau2, i, j] += FG1[q_ind, k_ind, 0, 1, tau1, i]*FG2[q_ind, k_ind, 1, 0, tau2, j]/(ek[k_ind, 0] - ekmq[q_ind, k_ind, 1]) + FG1[q_ind, k_ind, 0, 1, tau2, j]*FG2[q_ind, k_ind, 1, 0, tau1, i]/(ek[k_ind, 0] - ekmq[q_ind, k_ind, 1])
                        DG[tau1, tau2, i, j] += FE1[q_ind, k_ind, 0, 1, tau1, i]*FG2[q_ind, k_ind, 1, 0, tau2, j]/(ek[k_ind, 0] - ekmq[q_ind, k_ind, 1]) + FE1[q_ind, k_ind, 0, 1, tau2, j]*FG2[q_ind, k_ind, 1, 0, tau1, i]/(ek[k_ind, 0] - ekmq[q_ind, k_ind, 1])
                        DG[tau1, tau2, i, j] += FG1[q_ind, k_ind, 0, 1, tau1, i]*FE2[q_ind, k_ind, 1, 0, tau2, j]/(ek[ k_ind, 0] - ekmq[q_ind, k_ind, 1]) + FG1[q_ind, k_ind, 0, 1, tau2, j]*FE2[q_ind, k_ind, 1, 0, tau1, i]/(ek[k_ind, 0] - ekmq[q_ind, k_ind, 1])
                        if np.isnan(DG[tau,tau1, i, j]):
                            print('Something\'s NaN')
                            break
                        if tau1 == tau2:
                            for taux in range(2):
                                DG[tau1, tau1, i, j] += MG1[k_ind, tau1, taux, i, j]*pk[k_ind, 0, taux, tau1]
                                DG[tau1, tau1, i, j] += MEG1[k_ind, tau1, taux, i, j]*pk[k_ind, 0, taux, tau1]
                        DG[tau1, tau2, i, j] += -MG2[q_ind, k_ind, tau1, taux, i, j]*pk[k_ind, 0, tau2, tau1]
                        DG[tau1, tau2, i, j] += -MEG2[q_ind, k_ind, tau1, tau2, i, j]*pk[k_ind, 0, tau2, tau1]
    DG = DG/len(fbz)/M
    '''
    #convert to J^2
    DG = hbar**2*DG
    #convert to eV^2
    DG = DG/eVtoJ**2
    '''
    Dgeo.append(DG)
    print('%.2f'%(q_ind/len(qq)))

if save_data:
    np.save('c:/users/gugli/desktop/tesi/data/Dgeo.npy', Dgeo)

print('Time for calculation: %.1f min\n'%((time()-start)/60))
print('Total computation time: %.1f min\n'%((time()-timezero)/60))



