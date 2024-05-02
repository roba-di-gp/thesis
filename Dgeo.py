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

##values
print('Defining/importing numerical values')
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

#best fit parameters for phonon lines [meV^2]
pp_ph = np.load(r'c:/users/gugli/desktop/tesi/data/phonons_bestfit.npy')

#gamma parameters
gamma = np.load(r'c:/users/gugli/desktop/tesi/data/gamma.npy')
gamma_mat = np.array([np.array([gamma[0], gamma[1]]), np.array([gamma[1], gamma[0]])])

#FBZ meshgrid
fbz = np.load(r'c:/users/gugli/desktop/tesi/data/fbz_meshgrid.npy')

#differential
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

ticks = [0, 0.6666, 1., 1.5774]
ticklabels = ['$\Gamma$','K', 'M', '$\Gamma$']

for i in range(len(qind)):
    qind[i] = round(qind[i], 4)

qq = np.array([qqx, qqy])
qq = np.transpose(qq)

##evaluation
start = time()
#initialize variables with the right dimensions
#E(k), u(k), P(k)
ek = np.zeros((len(qq), len(fbz), 2)); uk = np.zeros((len(qq), len(fbz),2,2), dtype = 'complex'); pk = np.zeros((len(qq), len(fbz), 2, 2, 2), dtype = 'complex')
#E(-k), u(-k), P(-k)
emk = np.zeros((len(qq), len(fbz), 2)); umk = np.zeros((len(qq), len(fbz),2,2), dtype = 'complex'); pmk = np.zeros((len(qq), len(fbz), 2, 2, 2), dtype = 'complex')
#E(q-k), u(q-k), P(q-k)
eqmk = np.zeros((len(qq), len(fbz), 2)); uqmk = np.zeros((len(qq), len(fbz),2,2), dtype = 'complex'); pqmk = np.zeros((len(qq), len(fbz), 2, 2, 2), dtype = 'complex')
#E(k-q), u(k-q), P(k-q)
ekmq = np.zeros((len(qq), len(fbz), 2)); ukmq = np.zeros((len(qq), len(fbz),2,2), dtype = 'complex'); pkmq = np.zeros((len(qq), len(fbz), 2, 2, 2), dtype = 'complex')
#dE(-k), dP(-k)
demk = np.zeros((len(qq), len(fbz), 2, 2), dtype = 'complex'); dpmk = np.zeros((len(qq), len(fbz), 2, 2, 2, 2), dtype = 'complex')
#dE(q-k), dP(q-k)
deqmk = np.zeros((len(qq), len(fbz), 2, 2), dtype = 'complex'); dpqmk = np.zeros((len(qq), len(fbz), 2, 2, 2, 2), dtype = 'complex')
#ddE(-k), ddP(-k)
ddemk = np.zeros((len(qq), len(fbz), 2, 2, 2), dtype = 'complex'); ddpmk = np.zeros((len(qq), len(fbz), 2, 2, 2, 2, 2), dtype = 'complex')
#ddE(q-k), ddP(q-k)
ddeqmk = np.zeros((len(qq), len(fbz), 2, 2, 2), dtype = 'complex'); ddpqmk = np.zeros((len(qq), len(fbz), 2, 2, 2, 2, 2), dtype = 'complex')

print('Evaluating and vectorizing relevant quantities \n')
#cicle on the high-symmetry contour
for q_ind in range(len(qq)):
    q = qq[q_ind,:]
    #sum on the FBZ
    for k_ind in range(len(fbz)):
        k = fbz[k_ind,:]
        ek[q_ind, k_ind], uk[q_ind, k_ind], pk[q_ind, k_ind] = EUP(k, pp_el)
        emk[q_ind, k_ind], umk[q_ind, k_ind], pmk[q_ind, k_ind] = EUP(-k, pp_el)
        eqmk[q_ind, k_ind], uqmk[q_ind, k_ind], pqmk[q_ind, k_ind] = EUP(q-k, pp_el)
        ekmq[q_ind, k_ind], ukmq[q_ind, k_ind], pkmq[q_ind, k_ind] = EUP(k-q, pp_el)
        demk[q_ind, k_ind], dpmk[q_ind, k_ind] = dEdP(-k, pp_el, dk)
        deqmk[q_ind, k_ind], dpqmk[q_ind, k_ind] = dEdP(q-k, pp_el, dk)
        ddemk[q_ind, k_ind], ddpmk[q_ind, k_ind] = ddEddP(-k, pp_el, dk)
        ddeqmk[q_ind, k_ind], ddpqmk[q_ind, k_ind] = ddEddP(q-k, pp_el, dk)
        for tau1 in range(2):
            for tau2 in range(2):
                dpqmk[q_ind, k_ind, : , :, tau1, tau2] = gamma_mat[tau1, tau2]*dpqmk[q_ind, k_ind, : , :, tau1, tau2]
                dpmk[q_ind, k_ind] = gamma_mat[tau1, tau2]*dpmk[q_ind, k_ind, : , :, tau1, tau2]
                ddpqmk[q_ind, k_ind, : , :, tau1, tau2] = gamma_mat[tau1, tau2]*ddpqmk[q_ind, k_ind, : , :, tau1, tau2]
                ddpmk[q_ind, k_ind, : , :, :, tau1, tau2] = gamma_mat[tau1, tau2]*ddpmk[q_ind, k_ind, : , :, :, tau1, tau2]
    print('%.2f'%(q_ind/len(qq)))

print('Time for evaluation: %.1f s'%(time()-start))

##calculation of the matrix elements
start = time()
print('Calculating matrix elements\n')
DNG = np.zeros((2,2,2,2), dtype = 'complex')
#cicle on the high-symmetry contour
for q_ind in range(len(qq)):
    q = qq[q_ind,:]
    #sum on the FBZ
    for k_ind in range(len(fbz)):
        k = fbz[k_ind,:]
        #cicle on the dynamical matrix indices
        for tau in range(2):
            for tau1 in range(2):
                for i in range(2):
                    for j in range(2):
                        #sums on shared band index
                        for nu in range(2):
                            #terms involving first derivative
                            #sum on bands index
                            for nu1 in range(2):
                                #compute FG/FE element-wise
                                fg = np.zeros((2,2), dtype = 'complex')
                                fe = np.zeros((2,2), dtype = 'complex')
                                #fg indices
                                for tau0 in range(2):
                                    for i0 in range(2):
                                        #summed indices
                                        for nu2 in range(2):
                                            for tau2 in range(2):
                                                fg[tau0,i0] +=  np.transpose(np.conj(1j*eqmk[q_ind, k_ind, nu2]*np.conj(np.transpose(uk[q_ind, k_ind, nu, tau0]))*dpqmk[q_ind, k_ind,i0 , nu2, tau0, tau2]*ukmq[q_ind, k_ind, nu1,tau2])) +  1j*emk[q_ind, k_ind, nu2]*np.conj(np.transpose(ukmq[q_ind, k_ind, nu1, tau0]))*dpmk[q_ind, k_ind ,i0, nu2, tau0, tau2]*uk[q_ind, k_ind, nu,tau2]
                                                fg[tau0,i0] +=  np.transpose(np.conj(1j*deqmk[q_ind, k_ind, i0, nu2]*np.conj(np.transpose(uk[q_ind, k_ind, nu, tau0]))*pqmk[q_ind, k_ind, nu2, tau0, tau2]*ukmq[q_ind, k_ind, nu1,tau2])) +  1j*demk[q_ind, k_ind, i0, nu2]*np.conj(np.transpose(ukmq[q_ind, k_ind, nu1, tau0]))*pmk[q_ind, k_ind, nu2, tau0, tau2]*uk[q_ind, k_ind, nu,tau2]
                                DNG[tau, tau1, i, j] += 2*len(fbz)/M*(np.transpose(np.conj(fg))[tau,i]*fg[tau1,j]+np.transpose(np.conj(fg))[tau1,j]*fg[tau,i])/(ek[nu]-ekmq[nu1])*nFD(nu,k,pp_el, T)*(1-nFD(nu1, k-q,pp_el, T))
                                DNG[tau, tau1, i, j] += 2*len(fbz)/M*(np.transpose(np.conj(fe))[tau,i]*fg[tau1,j]+np.transpose(np.conj(fe))[tau1,j]*fg[tau,i])/(ek[nu]-ekmq[nu1])*nFD(nu,k,pp_el, T)*(1-nFD(nu1, k-q,pp_el, T))
                                DNG[tau, tau1, i, j] += 2*len(fbz)/M*(np.transpose(np.conj(fg))[tau,i]*fe[tau1,j]+np.transpose(np.conj(fg))[tau1,j]*fe[tau,i])/(ek[nu]-ekmq[nu1])*nFD(nu,k,pp_el, T)*(1-nFD(nu1, k-q,pp_el, T))
                            #terms involving the second derivative
                            #compure MEG/ME element-wise

    print('%.2f'%(q_ind/len(qq)))

print('Time for calculation: %.1f s'%(time()-start))



