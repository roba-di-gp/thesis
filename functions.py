import numpy as np
from numpy.linalg import eigh, inv

#hbar [eV s]
hbar = 6.582119569e-16
#lattice parameter [hbar*c/eV]
lp = 2.466731e-10/1.97327e-7
#lattice step [hbar*c/eV]
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

##electrons
#hamiltonian eigenvalues and eigenvectors
#(wavevector k, NN hopping t1, NNN hopping t2)
#returns eigenenergies E(k), eigenvector U(k), projector P(k)
#E[band index] in eV
#U[band index, sublattice index]
#P[band index, sublattice index 1, sublattice index 2]
def EUP(k, pp_el):
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
    h = np.array([[e0+t1, t0+t2],
        [np.conj(t0+t2), e0+t1]])
    #overlap matrix
    S = np.array([[1+s1,s0+s2], [np.conj(s0+s2), 1+s1]])
    #change basis
    invS = inv(S)
    htilde = np.matmul(h, invS)
    #diagonalize
    e, utilde = eigh(htilde)
    #return original eigenvectors and projectors
    u = np.matmul(S, utilde)
    u[0] = u[0]/np.sqrt((u[0,0]**2+u[0,1]**2))
    u[1] = u[1]/np.sqrt((u[1,0]**2+u[1,1]**2))
    p = np.zeros((2,2,2), dtype = 'complex')
    p[0] = np.outer(u[0], np.conj(np.transpose(u[0])))
    p[1] = np.outer(u[1], np.conj(np.transpose(u[1])))/(u[1,0]**2+u[1,1]**2)
    return e, u, p

#group velocity and projector derivatives
#dE[xy index, band index] in eV*m
#dP[xy index, band index, sublattice index1, sublattice index 2] in m
def dEdP(k, pp_el, dk):
    exp, null, pxp = EUP([k[0] + dk[0], k[1]], pp_el)
    exm, null, pxm = EUP([k[0] - dk[0], k[1]], pp_el)
    eyp, null, pyp = EUP([k[0], k[1] + dk[1]], pp_el)
    eym, null, pym = EUP([k[0], k[1] - dk[1]], pp_el)
    dedkx = (exp-exm)/2/dk[0]
    dedky = (eyp-eym)/2/dk[1]
    dpdkx = (pxp-pxm)/2/dk[0]
    dpdky = (pxp-pxm)/2/dk[1]
    dEdk = a*np.array([dedkx, dedky])
    dPdk = a*np.array([dpdkx, dpdky])
    return dEdk, dPdk

#mass^-1 tensor and second order projector derivatives
#ddE[xy index 1, xy index 2, band index] in eV*m^2
#ddP[xy index 1, xy index 2, band index, sublattice index 1, sublattice index 2] in m^2
def ddEddP(k, pp_el, dk):
    e, null, p = EUP([k[0], k[1]], pp_el)
    exp, null, pxp = EUP([k[0] + dk[0], k[1]], pp_el)
    exm, null, pxm = EUP([k[0] - dk[0], k[1]], pp_el)
    eyp, null, pyp = EUP([k[0], k[1] + dk[1]], pp_el)
    eym, null, pym = EUP([k[0], k[1] - dk[1]], pp_el)
    expyp, null, pxpyp = EUP([k[0] + dk[0], k[1]+dk[1]], pp_el)
    expym, null, pxpym = EUP([k[0] + dk[0], k[1]-dk[1]], pp_el)
    exmyp, null, pxmyp = EUP([k[0] - dk[0], k[1]+dk[1]], pp_el)
    exmym, null, pxmym = EUP([k[0] - dk[0], k[1]-dk[1]], pp_el)
    dexx = (exp - 2*e + exm)/dk[0]**2
    deyy = (eyp - 2*e + eym)/dk[1]**2
    dexy = (expyp - expym - exmyp + exmym)/(4*dk[0]*dk[1])
    dpxx = (pxp - 2*p + pxm)/dk[0]**2
    dpyy = (pyp - 2*p + pym)/dk[1]**2
    dpxy = (pxpyp - pxpym - pxmyp + pxmym)/(4*dk[0]*dk[1])
    ddEddk = a**2*np.array([[dexx, dexy], [dexy, deyy]])
    ddPddk = a**2*np.array([[dpxx, dpxy], [np.conj(dpxy), dpyy]])
    return ddEddk, ddPddk

##phonons
#flexural phonon dispersion lines
#(wavevector q, force constants A)
#returns 2 eigenfrequencies/eigenenergies
def ph_z(q, pp_ph):
    f1 = 0; f2 = 0
    for i in range(len(d0)):
        f1 = f1 + np.exp(1j*(q[0]*d0[i][0] + q[1]*d0[i][1]))
    for i in range(len(d1)):
        f2 = f2 + np.exp(1j*(q[0]*d1[i][0] + q[1]*d1[i][1]))
    a0 = -3*pp_ph[0] -6*pp_ph[1]
    a1 = f1*pp_ph[0]
    a2 = f2*pp_ph[1]
    #hamiltonian matrix
    h = np.array([[a0+a2, a1],
        [np.conj(a1), a0+a2]])
    w2, null = eigh(h)
    return np.sqrt(w2)

#in-plane phonon dispersion lines
#(wavevector q, force constants A)
#returns 4 eigenfrequencies/eigenenergies
def ph_xy(q, pp_ph):
    alpha = pp_ph[2]; beta = pp_ph[3];
    gamma = pp_ph[4]; delta = pp_ph[5]
    a = 2*gamma*(np.cos(np.sqrt(3)*q[1]) + 2*np.cos(3/2*q[0])*np.cos(np.sqrt(3)/2*q[1])-3)-3*alpha
    b = delta*(2*np.cos(np.sqrt(3)*q[1]) +2*np.cos(3/2*q[0]+2*np.pi/3)*np.exp(-1j*np.sqrt(3)/2*q[1]) +2*np.cos(3/2*q[0]-2*np.pi/3)*np.exp(1j*np.sqrt(3)/2*q[1]))
    c = alpha*(np.exp(1j*q[0]) +2*np.exp(-1j/2*q[0])*np.cos(np.sqrt(3)/2*q[1]) )
    d = beta*(np.exp(1j*q[0]) + 2*np.exp(-1j*q[0]/2)*np.cos(np.sqrt(3)/2*q[1]-2*np.pi/3))
    e = beta*(np.exp(1j*q[0]) + 2*np.exp(-1j*q[0]/2)*np.cos(np.sqrt(3)/2*q[1]+2*np.pi/3))
    hAA = np.array([[a,b], [np.conj(b), a]])
    hAB = np.array([[c,d],[e,c]])
    h = np.block([[hAA, hAB], [np.conj(np.transpose(hAB)), hAA]])
    w2, null = eigh(h)
    return np.sqrt(w2)

#basis change matrix
T = 1/np.sqrt(2)*np.array([np.array([1,1j,0,0]), np.array([1,-1j,0,0]), np.array([0,0,1,1j]), np.array([0,0,1,-1j])])

#in-plane phonon dispersion lines, modified to remove geometry
#(wavevector q, force constants pp_ph, geometric dynamical matrix Dgeo)
#returns 4 eigenfrequencies/eigenenergies
def ph_xy_NG(q, pp_ph, Dgeo):
    alpha = pp_ph[2]; beta = pp_ph[3];
    gamma = pp_ph[4]; delta = pp_ph[5]
    a = 2*gamma*(np.cos(np.sqrt(3)*q[1]) + 2*np.cos(3/2*q[0])*np.cos(np.sqrt(3)/2*q[1])-3)-3*alpha
    b = delta*(2*np.cos(np.sqrt(3)*q[1]) +2*np.cos(3/2*q[0]+2*np.pi/3)*np.exp(-1j*np.sqrt(3)/2*q[1]) +2*np.cos(3/2*q[0]-2*np.pi/3)*np.exp(1j*np.sqrt(3)/2*q[1]))
    c = alpha*(np.exp(1j*q[0]) +2*np.exp(-1j/2*q[0])*np.cos(np.sqrt(3)/2*q[1]) )
    d = beta*(np.exp(1j*q[0]) + 2*np.exp(-1j*q[0]/2)*np.cos(np.sqrt(3)/2*q[1]-2*np.pi/3))
    e = beta*(np.exp(1j*q[0]) + 2*np.exp(-1j*q[0]/2)*np.cos(np.sqrt(3)/2*q[1]+2*np.pi/3))
    hAA = np.array([[a,b], [np.conj(b), a]])
    hAB = np.array([[c,d],[e,c]])
    h = np.block([[hAA, hAB], [np.conj(np.transpose(hAB)), hAA]])
    h = np.matmul(np.matmul(T, h), np.transpose(np.conj(T)))
    h = h - Dgeo
    w2, null = eigh(h)
    return np.sqrt(w2)

##misc
#Fermi-Dirac counting factor
#(band index, wavevector, electron parameters, temperature)
#chemical potential is set to zero
def nFD(nu, k, pp_el, T):
    E, null, null = EUP(k, pp_el)
    n = 1/(1+np.exp((E[nu]-pp_el[0])/T))
    return n

#(band index 1, wavevector 1, band index 2, wavevector 2, electron parameters, differential, gamma parameters matrix)
#returns FE[sublattice index, cartesian index]
def fe(nu, k, nu1, k1, pp_el, dk, gamma_mat):
    #compute the relevant quantities
    null, u, null = EUP(k, pp_el)
    null, null, p1 = EUP(-k1, pp_el)
    null, u1, null = EUP(k1, pp_el)
    de1, null = dEdP(-k1, pp_el, dk)
    p1 = gamma_mat*p1
    #compute FE element-wise
    fe = np.zeros((2,2), dtype = 'complex')
    for tau in range(2):
        for i in range(2):
            for nu2 in range(2):
                for tau2 in range(2):
                    fe[tau,i] +=  1j*de1[i,nu2]*np.conj(np.transpose(u[nu, tau]))*p1[nu2, tau, tau2]*u1[nu1,tau2]
    return fe

def FE(nu, k, nu1, k1, pp_el, dk, gamma_mat):
    fe01 = fe(nu, k, nu1, k1, pp_el, dk, gamma_mat)
    fe10 = fe(nu1, k1, nu, k, pp_el, dk, gamma_mat)
    FE = np.transpose(np.conj(fe01)) + fe10
    return FE

#(band index 1, wavevector 1, band index 2, wavevector 2, electron parameters, differential, gamma parameters matrix)
#returns FG[sublattice index, cartesian index]
def fg(nu, k, nu1, k1, pp_el, dk, gamma_mat):
    #compute the relevant quantities
    null, u, null = EUP(k, pp_el)
    e1, null, null = EUP(-k1, pp_el)
    null, u1, null = EUP(k1, pp_el)
    null, dp1 = dEdP(-k1, pp_el, dk)
    dp1 = gamma_mat*dp1
    #compute FE element-wise
    fg = np.zeros((2,2), dtype = 'complex')
    for tau in range(2):
        for i in range(2):
            for nu2 in range(2):
                for tau2 in range(2):
                    fg[tau,i] +=  1j*e1[nu2]*np.conj(np.transpose(u[nu, tau]))*dp1[i, nu2, tau, tau2]*u1[nu1,tau2]
    return fg

def FG(nu, k, nu1, k1, pp_el, dk, gamma_mat):
    fg01 = fg(nu, k, nu1, k1, pp_el, dk, gamma_mat)
    fg10 = fg(nu1, k1, nu, k, pp_el, dk, gamma_mat)
    FG = np.transpose(np.conj(fg01)) + fg10
    return FG

#(band index 1, wavevector 1, wavevector 2, electron parameters, differential, gamma parameters matrix)
#returns MG[sublattice index, cartesian index 1, cartesian index 2]
def MG(nu, k, k1, pp_el, dk, gamma_mat):
    null, null, p = EUP(k, pp_el)
    e1, null, null = EUP(k1, pp_el)
    null, ddp1 = ddEddP(k1, pp_el, dk)
    ddp1 = gamma_mat**2*ddp1
    MG = np.zeros((2,2,2), dtype = 'complex')
    for tau in range(2):
        for i in range(2):
            for j in range(2):
                for nu2 in range(2):
                    for tau2 in range(2):
                        MG[tau, i, j] += -e1[nu2]*ddp1[i,j,tau,tau2]*p[nu,tau2, tau]
    return MG

#(band index 1, wavevector 1, wavevector 2, electron parameters, differential, gamma parameters matrix)
#returns MEG[sublattice index, cartesian index 1, cartesian index 2]
def MEG(nu, k, k1, pp_el, dk, gamma_mat):
    null, null, p = EUP(k, pp_el)
    de1, dp1 = dEdP(k1, pp_el)
    dp1 = gamma_mat**2*dp1
    MEG = np.zeros((2,2,2), dtype = 'complex')
    for tau in range(2):
        for i in range(2):
            for j in range(2):
                for nu2 in range(2):
                    for tau2 in range(2):
                        MEG[tau,i,j] += -de1[i, nu2]*dp1[j,nu2,tau,tau2]*p[nu,tau2,tau]
    return MEG

