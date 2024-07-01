import numpy as np
from numpy.linalg import eigh, inv
import matplotlib.pyplot as plt

#lattice parameter (from Guido) [nm]
lp = 0.2466731
#lattice step [nm]
a = lp/np.sqrt(3)

#NN vectors [in units of 1/a]
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

##electrons
#hamiltonian eigenvalues and eigenvectors
#(wavevector k, electron parameters, gap)
#returns eigenenergies E(k), eigenvector U(k), projector P(k)
#E[band index] in eV
#U[band index, sublattice index]
#P[band index, sublattice index 1, sublattice index 2]
def EUP(k, pp_el, gap = 0):
    e0 = pp_el[0]; t = pp_el[1:4]; s = pp_el[4:7]
    #initialize values
    t0 = 0; t1 = 0; t2 = 0; s0 = 0; s1 = 0; s2 = 0
    #NN geometric hopping factor
    for i in range(len(d0)):
        t0 = t0 + t[0]*np.exp(-1j*k[0]*d0[i][0] - 1j*k[1]*d0[i][1])
        s0 = s0 + s[0]*np.exp(-1j*k[0]*d0[i][0] - 1j*k[1]*d0[i][1])
    #NNN geometric hopping factor
    for i in range(len(d1)):
        t1 = t1 + t[1]*np.exp(-1j*k[0]*d1[i][0] - 1j*k[1]*d1[i][1])
        s1 = s1 + s[1]*np.exp(-1j*k[0]*d1[i][0] - 1j*k[1]*d1[i][1])
    #NNNN geometric hopping factor
    for i in range(len(d2)):
        t2 = t2 + t[2]*np.exp(-1j*k[0]*d2[i][0] - 1j*k[1]*d2[i][1])
        s2 = s2 + s[2]*np.exp(-1j*k[0]*d2[i][0] - 1j*k[1]*d2[i][1])
    #hamiltonian matrix
    h = np.array([[e0+gap/2+t1, t0+t2],
        [np.conj(t0+t2), e0-gap/2+t1]])
    #overlap matrix
    S = np.array([[1+s1, s0+s2], [np.conj(s0+s2), 1+s1]])
    #change basis
    invS = inv(S)
    htilde = np.matmul(h, invS)
    #diagonalize
    e, utilde = eigh(htilde)
    #valence eigenvector
    utildev = utilde[:,0]
    #conduction eigenvector
    utildec = utilde[:,1]
    #return original eigenvectors and projectors
    uv = np.matmul(invS, utildev)
    uc = np.matmul(invS, utildec)
    #normalize
    uv = uv/np.sqrt((abs(uv[0])**2+abs(uv[1])**2))
    uc = uc/np.sqrt((abs(uc[0])**2+abs(uc[1])**2))
    u = np.array([uv, uc])
    #define projectors
    p = np.zeros((2,2,2), dtype = 'complex')
    p[0] = np.outer(uv, np.conj(uv))
    p[1] = np.outer(uc, np.conj(uc))
    return e, u, p

#group velocity and projector derivatives
#dE[xy index, band index] in eV*nm
#dP[xy index, band index, sublattice index1, sublattice index 2] in nm
def dEdP(k, pp_el, dk, gap = 0.01):
    exp, null, pxp = EUP([k[0] + dk[0], k[1]], pp_el, gap)
    exm, null, pxm = EUP([k[0] - dk[0], k[1]], pp_el, gap)
    eyp, null, pyp = EUP([k[0], k[1] + dk[1]], pp_el, gap)
    eym, null, pym = EUP([k[0], k[1] - dk[1]], pp_el, gap)
    dedkx = (exp-exm)/2/dk[0]
    dedky = (eyp-eym)/2/dk[1]
    dpdkx = (pxp-pxm)/2/dk[0]
    dpdky = (pyp-pym)/2/dk[1]
    dEdk = a*np.array([dedkx, dedky])
    dPdk = a*np.array([dpdkx, dpdky])
    return dEdk, dPdk

#mass^-1 tensor and second order projector derivatives
#ddE[xy index 1, xy index 2, band index] in eV*nm^2
#ddP[xy index 1, xy index 2, band index, sublattice index 1, sublattice index 2] in nm^2
def ddEddP(k, pp_el, dk, gap = 0.01):
    e, null, p = EUP([k[0], k[1]], pp_el, gap)
    exp, null, pxp = EUP([k[0] + dk[0], k[1]], pp_el, gap)
    exm, null, pxm = EUP([k[0] - dk[0], k[1]], pp_el, gap)
    eyp, null, pyp = EUP([k[0], k[1] + dk[1]], pp_el, gap)
    eym, null, pym = EUP([k[0], k[1] - dk[1]], pp_el, gap)
    expyp, null, pxpyp = EUP([k[0] + dk[0], k[1]+dk[1]], pp_el, gap)
    expym, null, pxpym = EUP([k[0] + dk[0], k[1]-dk[1]], pp_el, gap)
    exmyp, null, pxmyp = EUP([k[0] - dk[0], k[1]+dk[1]], pp_el, gap)
    exmym, null, pxmym = EUP([k[0] - dk[0], k[1]-dk[1]], pp_el, gap)
    dexx = (exp - 2*e + exm)/dk[0]**2
    deyy = (eyp - 2*e + eym)/dk[1]**2
    dexy = (expyp - expym - exmyp + exmym)/(4*dk[0]*dk[1])
    dpxx = (pxp - 2*p + pxm)/dk[0]**2
    dpyy = (pyp - 2*p + pym)/dk[1]**2
    dpxy = (pxpyp - pxpym - pxmyp + pxmym)/(4*dk[0]*dk[1])
    ddEddk = a**2*np.array([[dexx, dexy], [dexy, deyy]])
    ddPddk = a**2*np.array([[dpxx, dpxy], [dpxy, dpyy]])
    return ddEddk, ddPddk

##phonons
#flexural phonon dispersion lines
#(wavevector q, force constants A)
#returns 2 eigenfrequencies/eigenenergies
def ph_z(q, pp_ph_z):
    f1 = 0; f2 = 0
    for i in range(len(d0)):
        f1 = f1 + np.exp(1j*(q[0]*d0[i][0] + q[1]*d0[i][1]))
    for i in range(len(d1)):
        f2 = f2 + np.exp(1j*(q[0]*d1[i][0] + q[1]*d1[i][1]))
    a0 = - 3*pp_ph_z[0] - 6*pp_ph_z[1]
    a1 = f1*pp_ph_z[0]
    a2 = f2*pp_ph_z[1]
    #dynamical matrix
    h = np.array([[a0+a2, a1],
        [np.conj(a1), a0+a2]])
    w2, null = eigh(h)
    return np.sqrt(w2)

#in-plane phonon dispersion lines
#(wavevector q, force constants A)
#returns 4 eigenfrequencies/eigenenergies
def ph_xy(q, pp_ph_xy):
    alpha = pp_ph_xy[0]; beta = pp_ph_xy[1];
    gamma = pp_ph_xy[2]; delta = pp_ph_xy[3]
    a = 2*gamma*(np.cos(np.sqrt(3)*q[1]) + 2*np.cos(3/2*q[0])*np.cos(np.sqrt(3)/2*q[1])-3)-3*alpha
    b = delta*(2*np.cos(np.sqrt(3)*q[1]) +2*np.cos(3/2*q[0]+2*np.pi/3)*np.exp(-1j*np.sqrt(3)/2*q[1]) +2*np.cos(3/2*q[0]-2*np.pi/3)*np.exp(1j*np.sqrt(3)/2*q[1]))
    c = alpha*(np.exp(1j*q[0]) +2*np.exp(-1j/2*q[0])*np.cos(np.sqrt(3)/2*q[1]) )
    d = beta*(np.exp(1j*q[0]) + 2*np.exp(-1j*q[0]/2)*np.cos(np.sqrt(3)/2*q[1]-2*np.pi/3))
    e = beta*(np.exp(1j*q[0]) + 2*np.exp(-1j*q[0]/2)*np.cos(np.sqrt(3)/2*q[1]+2*np.pi/3))
    dAA = np.array([[a,b], [np.conj(b), a]])
    dAB = np.array([[c,d],[e,c]])
    D = np.block([[dAA, dAB], [np.conj(np.transpose(dAB)), dAA]])
    w2, null = eigh(D)
    return np.sqrt(w2)

#squared flexural phonon dispersion lines
#(wavevector q, force constants A)
#returns 2 squared eigenfrequencies/eigenenergies
def ph_z_sq(q, pp_ph_z):
    f1 = 0; f2 = 0
    for i in range(len(d0)):
        f1 = f1 + np.exp(1j*(q[0]*d0[i][0] + q[1]*d0[i][1]))
    for i in range(len(d1)):
        f2 = f2 + np.exp(1j*(q[0]*d1[i][0] + q[1]*d1[i][1]))
    a0 = - 3*pp_ph_z[0] - 6*pp_ph_z[1]
    a1 = f1*pp_ph_z[0]
    a2 = f2*pp_ph_z[1]
    #dynamical matrix
    h = np.array([[a0+a2, a1],
        [np.conj(a1), a0+a2]])
    w2, null = eigh(h)
    return w2

#squared in-plane phonon dispersion lines
#(wavevector q, force constants A)
#returns 4 squared eigenfrequencies/eigenenergies
def ph_xy_sq(q, pp_ph_xy):
    alpha = pp_ph_xy[0]; beta = pp_ph_xy[1];
    gamma = pp_ph_xy[2]; delta = pp_ph_xy[3]
    a = 2*gamma*(np.cos(np.sqrt(3)*q[1]) + 2*np.cos(3/2*q[0])*np.cos(np.sqrt(3)/2*q[1])-3)-3*alpha
    b = delta*(2*np.cos(np.sqrt(3)*q[1]) +2*np.cos(3/2*q[0]+2*np.pi/3)*np.exp(-1j*np.sqrt(3)/2*q[1]) +2*np.cos(3/2*q[0]-2*np.pi/3)*np.exp(1j*np.sqrt(3)/2*q[1]))
    c = alpha*(np.exp(1j*q[0]) +2*np.exp(-1j/2*q[0])*np.cos(np.sqrt(3)/2*q[1]) )
    d = beta*(np.exp(1j*q[0]) + 2*np.exp(-1j*q[0]/2)*np.cos(np.sqrt(3)/2*q[1]-2*np.pi/3))
    e = beta*(np.exp(1j*q[0]) + 2*np.exp(-1j*q[0]/2)*np.cos(np.sqrt(3)/2*q[1]+2*np.pi/3))
    dAA = np.array([[a,b], [np.conj(b), a]])
    dAB = np.array([[c,d],[e,c]])
    D = np.block([[dAA, dAB], [np.conj(np.transpose(dAB)), dAA]])
    w2, null = eigh(D)
    return w2

#basis change matrix
T = 1/np.sqrt(2)*np.array([[1,1j,0,0], [1,-1j,0,0], [0,0,1,1j], [0,0,1,-1j]])

#in-plane phonon dispersion lines, modified to remove geometry
#(wavevector q, force constants pp_ph, geometric dynamical matrix Dgeo)
#returns 4 eigenfrequencies/eigenenergies
def ph_xy_NG(q, pp_ph_xy, DGeo):
    alpha = pp_ph_xy[0]; beta = pp_ph_xy[1];
    gamma = pp_ph_xy[2]; delta = pp_ph_xy[3]
    a = 2*gamma*(np.cos(np.sqrt(3)*q[1]) + 2*np.cos(3/2*q[0])*np.cos(np.sqrt(3)/2*q[1])-3)-3*alpha
    b = delta*(2*np.cos(np.sqrt(3)*q[1]) +2*np.cos(3/2*q[0]+2*np.pi/3)*np.exp(-1j*np.sqrt(3)/2*q[1]) +2*np.cos(3/2*q[0]-2*np.pi/3)*np.exp(1j*np.sqrt(3)/2*q[1]))
    c = alpha*(np.exp(1j*q[0]) +2*np.exp(-1j/2*q[0])*np.cos(np.sqrt(3)/2*q[1]) )
    d = beta*(np.exp(1j*q[0]) + 2*np.exp(-1j*q[0]/2)*np.cos(np.sqrt(3)/2*q[1]-2*np.pi/3))
    e = beta*(np.exp(1j*q[0]) + 2*np.exp(-1j*q[0]/2)*np.cos(np.sqrt(3)/2*q[1]+2*np.pi/3))
    dAA = np.array([[a,b], [np.conj(b), a]])
    dAB = np.array([[c,d],[e,c]])
    D = np.block([[dAA, dAB], [np.conj(np.transpose(dAB)), dAA]])
    D = np.matmul(np.matmul(T, D), np.transpose(np.conj(T)))
    DNG = np.zeros((4,4), dtype='complex')
    DNG = D - DGeo
    w2, null = eigh(DNG)
    return np.sqrt(w2)

##fbz meshgrid
def fbz_meshgrid(N, S = 1, q0 = np.zeros(2)):
    #build a NxN meshgrid on the BZ scaled by a factor S and centered in q0
    #wavevectors are in units of 1/a
    #primitive vectors
    b1 = np.array([2*np.pi/3, 2*np.pi/np.sqrt(3)])
    b2 = np.array([2*np.pi/3, -2*np.pi/np.sqrt(3)])

    #build the grid on the PC
    kpc = []
    for i in range(N):
        for j in range(N):
            kpc.append(i/N*b1 + j/N*b2)

    #fold on the FBZ
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
    k = S*k + q0
    return k

def ext_fbz_meshgrid(N, S=1, q0 = np.array([0,0])):
    #build a 6 x N x N hexagonal meshgrid centered in q0, rescaled by S
    #wavevectors are in units of 1/a
    #primitive vectors
    b1 = np.array([2*np.pi/3, 2*np.pi/np.sqrt(3)])
    b2 = np.array([2*np.pi/3, -2*np.pi/np.sqrt(3)])

    #build the grid on the PC
    kpc = []
    for i in range(N):
        for j in range(N):
            kpc.append(i/N*b1 + j/N*b2)

    #fold on the FBZ
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

    kext = np.concatenate((k, k + b1), axis = 0)
    kext = np.concatenate((kext, k + b2), axis = 0)
    kext = np.concatenate((kext, k - b1), axis = 0)
    kext = np.concatenate((kext, k - b2), axis = 0)
    kext = np.concatenate((kext, k + b1 + b2), axis = 0)
    kext = np.concatenate((kext, k - b1 - b2), axis = 0)
    kext = kext*S + q0
    return kext

##quantum geometric dynamical matrix
#calculates the geometric contribution to dynamical matrix elements for a certain phonon wavevector q
#(phonon wavevector q [1/a], FBZ meshgrid [1/a], electron parameters [eV], gamma parameters [nm^-2], differential [a], Dirac point gap [eV])
#returns a 4x4 matrix in eV^2
def calc_DGeo(q, fbzmesh, pp_el, gamma_mat, our_dk, gapp = 0.001):
    DGeo = np.zeros((2,2,2,2), dtype = 'complex')
    #regularizer [eV], to be set to zero if gapp != 0
    eta = 0
    #sum on the FBZ meshgrid
    for k in fbzmesh:
        ek, uk, pk = EUP(k, pp_el, gapp)
        emk, umk, pmk = EUP(-k, pp_el, gapp)
        dek, dpk = dEdP(k, pp_el, our_dk, gapp)
        demk, dpmk = dEdP(-k, pp_el, our_dk, gapp)
        ddek, ddpk = ddEddP(k, pp_el, our_dk, gapp)
        ekmq, ukmq, pkmq = EUP(k-q, pp_el, gapp)
        ekpq, ukpq, pkpq = EUP(k+q, pp_el, gapp)
        eqmk, uqmk, pqmk = EUP(q-k, pp_el, gapp)
        emkmq, umkmq, pmkmq = EUP(-k-q, pp_el, gapp)
        dekmq, dpkmq = dEdP(k-q, pp_el, our_dk, gapp)
        dekpq, dpkpq = dEdP(k+q, pp_el, our_dk, gapp)
        demkmq, dpmkmq = dEdP(-k-q, pp_el, our_dk, gapp)
        deqmk, dpqmk = dEdP(q-k, pp_el, our_dk, gapp)
        null, ddpkpq = ddEddP(k+q, pp_el, our_dk, gapp)
        #initialize F tensors [tau, i]
        fEkkmq = np.zeros((2,2), dtype = 'complex')
        fGkkmq = np.zeros((2,2), dtype = 'complex')
        fEkmqk = np.zeros((2,2), dtype = 'complex')
        fGkmqk = np.zeros((2,2), dtype = 'complex')
        fEkkpq = np.zeros((2,2), dtype = 'complex')
        fGkkpq = np.zeros((2,2), dtype = 'complex')
        fEkpqk = np.zeros((2,2), dtype = 'complex')
        fGkpqk = np.zeros((2,2), dtype = 'complex')
        #initialize m tensors [tau1, tau2, i, j]
        mEGk = np.zeros((2,2,2,2), dtype = 'complex')
        mEGkpq = np.zeros((2,2,2,2), dtype = 'complex')
        mGk = np.zeros((2,2,2,2), dtype = 'complex')
        mGkpq = np.zeros((2,2,2,2), dtype = 'complex')
        #cycle on F tensors indices
        for tau in range(2):
            for i in range(2):
                #sum on mute indices
                for taux in range(2):
                    for nux in range(2):
                        fEkkmq[tau, i] += 1j*gamma_mat[tau, taux]*(dek[i, nux]*ukmq[1, tau]*pk[nux, tau, taux]*np.conj(uk[0, taux]) + deqmk[i, nux]*np.conj(uk[0, tau])*pqmk[nux, tau, taux]*ukmq[1, taux])
                        fEkmqk[tau, i] += 1j*gamma_mat[tau, taux]*(dekmq[i, nux]*uk[0, tau]*pkmq[ nux, tau, taux]*np.conj(ukmq[ 1, taux]) + demk[i, nux]*np.conj(ukmq[1, tau])*pmk[ nux, tau, taux]*uk[0, taux])
                        fEkkpq[tau, i] += 1j*gamma_mat[tau, taux]*(dek[i, nux]*ukpq[1, tau]*pk[nux, tau, taux]*np.conj(uk[0, taux]) + demkmq[i, nux]*np.conj(uk[0, tau])*pmkmq[nux, tau, taux]*ukpq[1, taux])
                        fEkpqk[tau, i] += 1j*gamma_mat[tau, taux]*(dekpq[i, nux]*uk[0, tau]*pkpq[nux, tau, taux]*np.conj(ukpq[1, taux]) + demk[i, nux]*np.conj(ukpq[1, tau])*pmk[nux, tau, taux]*uk[0, taux])
                        fGkkmq[tau, i] += 1j*gamma_mat[tau, taux]*(ek[nux]*ukmq[1, tau]*dpk[i, nux, tau, taux]*np.conj(uk[0, taux]) + eqmk[nux]*np.conj(uk[0, tau])*dpqmk[i, nux, tau, taux]*ukmq[1, taux])
                        fGkmqk[tau, i] += 1j*gamma_mat[tau, taux]*(ekmq[ nux]*uk[0, tau]*dpkmq[i, nux, tau, taux]*np.conj(ukmq[1, taux]) + emk[nux]*np.conj(ukmq[1, tau])*dpmk[i, nux, tau, taux]*uk[0, taux])
                        fGkkpq[tau, i] += 1j*gamma_mat[tau, taux]*(ek[nux]*ukpq[1, tau]*dpk[i, nux, tau, taux]*np.conj(uk[0, taux]) + emkmq[nux]*np.conj(uk[0, tau])*dpmkmq[i, nux, tau, taux]*ukpq[1, taux])
                        fGkpqk[tau, i] += 1j*gamma_mat[tau, taux]*(ekpq[nux]*uk[0, tau]*dpkpq[i, nux, tau, taux]*np.conj(ukpq[1, taux]) + emk[nux]*np.conj(ukpq[1, tau])*dpmk[i, nux, tau, taux]*uk[0, taux])
        #cycle on M tensors indices
        for tau1 in range(2):
            for tau2 in range(2):
                for i in range(2):
                    for j in range(2):
                        #sum on mute index
                        for nux in range(2):
                            mEGk[tau1, tau2, i, j] += -gamma_mat[tau1, tau2]**2*(dek[i, nux]*dpk[j, nux, tau1, tau2] + dek[j, nux]*dpk[i, nux, tau1, tau2])
                            mGk[tau1, tau2, i, j] += -gamma_mat[tau1, tau2]**2*ek[nux]*ddpk[i,j,nux, tau1, tau2]
                            mEGkpq[tau1, tau2, i, j] += -gamma_mat[tau1, tau2]**2*(dekpq[i, nux]*dpkpq[j, nux, tau1, tau2] + dekpq[j, nux]*dpkpq[i, nux, tau1, tau2])
                            mGkpq[ tau1, tau2, i, j] += -gamma_mat[tau1, tau2]**2*ekpq[nux]*ddpkpq[i, j, nux, tau1, tau2]
        #cycle on DM indices
        for tau1 in range(2):
            for tau2 in range(2):
                for i in range(2):
                    for j in range(2):
                        DGeo[tau1, tau2, i, j] += fGkkmq[ tau1, i]*fGkmqk[ tau2, j]/np.sqrt((ek[0]-ekpq[1])**2 + eta**2)*np.sign(ek[0]-ekpq[1]) + fGkkpq[tau2, j]*fGkpqk[ tau1, i]/np.sqrt((ek[0]-ekmq[1])**2 + eta**2)*np.sign(ek[0]-ekmq[1])
                        DGeo[tau1, tau2, i, j] += fGkkmq[ tau1, i]*fEkmqk[ tau2, j]/np.sqrt((ek[0]-ekpq[1])**2 + eta**2)*np.sign(ek[0]-ekpq[1]) + fGkkpq[tau2, j]*fEkpqk[ tau1, i]/np.sqrt((ek[0]-ekmq[1])**2 + eta**2)*np.sign(ek[0]-ekmq[1])
                        DGeo[tau1, tau2, i, j] += fEkkmq[ tau1, i]*fGkmqk[ tau2, j]/np.sqrt((ek[0]-ekpq[1])**2 + eta**2)*np.sign(ek[0]-ekpq[1]) + fEkkpq[tau2, j]*fGkpqk[ tau1, i]/np.sqrt((ek[0]-ekmq[1])**2 + eta**2)*np.sign(ek[0]-ekmq[1])
                        DGeo[tau1, tau2, i, j] += - mEGkpq[tau1, tau2, i,j]*pk[0, tau2, tau1]
                        DGeo[tau1, tau2, i, j] += - mGkpq[tau1, tau2, i,j]*pk[0, tau2, tau1]
                        if (tau1 == tau2):
                            for taux in range(2):
                                DGeo[tau1, tau1, i, j] += mEGk[tau1, taux, i, j]*pk[0, taux, tau1]
                                DGeo[tau1, tau1, i, j] += mGk[tau1, taux, i, j]*pk[0, taux, tau1]
    #from 2x2x2x2 to 4x4
    DGeoblock = np.block([[DGeo[0,0], DGeo[0,1]], [DGeo[1,0],
DGeo[1,1]]])
    #1/M 1/N prefactor
    DGeoblock = DGeoblock/M/len(fbzmesh)
    #from fs^-2 to eV^2
    DGeoblock = hbar**2*DGeoblock
    return DGeoblock