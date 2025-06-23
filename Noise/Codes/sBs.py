import numpy as np
import qutip as qt
import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
from tqdm import tqdm
from joblib import Parallel, delayed
import os
from bits import *

#define operators
N = 140 #fix fock dimension
l = np.sqrt(2*np.pi)
I = sps.csc_matrix(np.eye(N))
a_op = qt.destroy(N)
n_op = a_op.dag()*a_op
a_op = sps.csc_matrix(a_op.full())
n_op = sps.csc_matrix(n_op.full())
q_op = (a_op + a_op.T)/np.sqrt(2)
p_op = -1j*(a_op - a_op.T)/np.sqrt(2)
ket_vacuum = sps.csc_matrix(qt.basis(N,0).full())
rho_vacuum = qt.ket2dm(qt.basis(N,0)).full()
rho_vacuum = sps.csc_matrix(rho_vacuum)
Tp = spsla.expm(1j*l*p_op)
Tq = spsla.expm(1j*l*q_op)
Dict_T = {'T1':Tq,'T2':Tp}

sigma_x = sps.csc_matrix(qt.sigmax().full())
sigma_z = sps.csc_matrix(qt.sigmaz().full())
ket_0 = qt.basis(2,1).full()
ket_1 = qt.basis(2,0).full()
rho_g = sps.csc_matrix(qt.ket2dm(qt.basis(2,0)).full())
rho_e = sps.csc_matrix(qt.ket2dm(qt.basis(2,1)).full())
ket_plus = sps.csc_matrix((ket_0 + ket_1)/np.sqrt(2))
ket_minus = sps.csc_matrix((ket_0 - ket_1)/np.sqrt(2))
rho_plus = sps.csc_matrix(qt.ket2dm(qt.basis(2,0)+qt.basis(2,1)).full())/2
rho_minus = sps.csc_matrix(qt.ket2dm(qt.basis(2,0)-qt.basis(2,1)).full())/2
I_ket_g = sps.kron(I,ket_0)
I_ket_e = sps.kron(I,ket_1)
I_ket_plus = sps.kron(I,ket_plus)

rho_vacuum_plus = sps.kron(rho_vacuum,ket_plus)
rho_vacuum_g = sps.kron(rho_vacuum,ket_0)
rho_vacuum_e = sps.kron(rho_vacuum,ket_1)


#basic functions
def Displacement(alpha):
    return sps.csc_matrix(spsla.expm(alpha*a_op.T - alpha.conjugate()*a_op))
def CD(alpha):
    return spsla.expm(sps.kron(alpha*a_op.T - alpha.conjugate()*a_op,sigma_z/2/np.sqrt(2)))
def Krauss_from_U(U):
    #U acts on the system and ancilla
    #assumme qubit starts on |+>
    Kg = sps.csc_matrix(I_ket_g.getH()@U@I_ket_plus)
    Ke = sps.csc_matrix(I_ket_e.getH()@U@I_ket_plus)
    Kg.data[np.abs(Kg.data) < 1e-7] = 0
    Ke.data[np.abs(Ke.data) < 1e-7] = 0
    Kg.eliminate_zeros()
    Ke.eliminate_zeros()
    return Kg.tocsc(), Ke.tocsc()

def stabilizers(Delta):
    E = spsla.expm(-Delta**2*n_op)
    E_inv = spsla.inv(E.tocsc())
    Tq, Tp = Dict_T['T1'], Dict_T['T2']
    return E@Tq@E_inv, E@Tp@E_inv

def displaced_stabilizers(Tq_delta, Tp_delta,alpha):
    # T a stabilizer in sp.csc_matrix form, alpha a complex number
    # returns displaced stabilizer
    D = Displacement(alpha)
    return D@Tq_delta@D.getH(), D@Tp_delta@D.getH()

def I_Rx(theta):
    Rx = np.array([[np.cos(theta/2),-1j*np.sin(theta/2)],[-1j*np.sin(theta/2),np.cos(theta/2)]])
    return sps.kron(I,sps.csc_matrix(Rx))


#sBs with reset
def sBs_p(mu,Delta):
    #corresponds to pg \simeq (1-nu\sin(lp))/2
    sd, cd, td = np.sinh(Delta**2), np.cosh(Delta**2), np.tanh(Delta**2)
    large, small = l*cd, 1j*l*sd/2
    nu = (-1)**mu
    return CD(small)@I_Rx(nu*np.pi/2)@CD(large)@I_Rx(np.pi/2)@CD(small)
def sBs_q(mu,Delta):
    #corresponds to pg \simeq (1-nu\sin(lq))/2
    sd, cd, td = np.sinh(Delta**2), np.cosh(Delta**2), np.tanh(Delta**2)
    large, small = -1j*l*cd, l*sd/2
    nu = (-1)**mu
    return CD(small)@I_Rx(-nu*np.pi/2)@CD(large)@I_Rx(np.pi/2)@CD(small)

def sBs_p_measurement(mu,Delta):
    #corresponds to pg \simeq (1-nu\sin(lp))/2
    sd, cd, td = np.sinh(Delta**2), np.cosh(Delta**2), np.tanh(Delta**2)
    large, small = l*cd, 1j*l*sd/2
    nu = (-1)**mu
    return I_Rx(nu*np.pi/2)@CD(large)@I_Rx(np.pi/2)@CD(small)
def sBs_q_measurement(mu,Delta):
    #corresponds to pg \simeq (1-nu\sin(lq))/2
    sd, cd, td = np.sinh(Delta**2), np.cosh(Delta**2), np.tanh(Delta**2)
    large, small = -1j*l*cd, l*sd/2
    nu = (-1)**mu
    return I_Rx(-nu*np.pi/2)@CD(large)@I_Rx(np.pi/2)@CD(small)

def Krauss_dictionaries(Delta):
    Uq0, Uq1  = sBs_q_measurement(0,Delta), sBs_q_measurement(1,Delta)
    Up0, Up1 = sBs_p_measurement(0,Delta), sBs_p_measurement(1,Delta)

    cd, sd = np.cosh(Delta**2), np.sinh(Delta**2)
    small_p = (-1j*l*sd/2)/2/np.sqrt(2)
    small_q = (l*sd/2)/2/np.sqrt(2)
    Uqe, Uqg = Displacement(small_q), Displacement(-small_q)
    Upe, Upg = Displacement(small_p), Displacement(-small_p)

    Kgq0, Keq0 = Krauss_from_U(Uq0)
    Kgq1, Keq1 = Krauss_from_U(Uq1)
    Kgp0, Kep0 = Krauss_from_U(Up0)
    Kgp1, Kep1 = Krauss_from_U(Up1)
    Krauss_dictionary = {'Kgq0':Kgq0,'Keq0':Keq0,'Kgq1':Kgq1,'Keq1':Keq1,'Kgp0':Kgp0,'Kep0':Kep0,'Kgp1':Kgp1,'Kep1':Kep1}
    Krauss_dictionary_U = {'Kgq0':Uqg @ Kgq0,'Keq0':Uqe @ Keq0,'Kgq1':Uqg @ Kgq1,'Keq1':Uqe @ Keq1,'Kgp0':Upg @ Kgp0,'Kep0':Upe @ Kep0,'Kgp1':Upg @ Kgp1,'Kep1':Upe @ Kep1}
    return Krauss_dictionary, Krauss_dictionary_U

def sBs_cycle(Delta,R,rho0,gauges):
    #R is the number of q-p cycles
    #rho0 the initial state of the oscillator 
    #returns all density matrices and probabilities. Length of the list is R+1.

    Krauss_dictionary, Krauss_dictionary_U = Krauss_dictionaries(Delta)
    rho = rho0.copy()
    rhos = [rho.copy()]
    mu_q, mu_p = gauges[0], gauges[1] #initialize the gauges, this assures final preparation of sensor in 0,0 when starting in vacuum. 
    probs_gq, probs_gp = [], []
    for i in range(R):
        Kgq, Keq = Krauss_dictionary_U[f'Kgq{mu_q}'], Krauss_dictionary_U[f'Keq{mu_q}']
        probs_gq.append((Kgq.getH()@Kgq@rho).diagonal().sum())
        rho = Kgq @ rho @ Kgq.getH() + Keq @rho @ Keq.getH()     
        mu_p = (mu_p +1)% 2#gauge update
        
        Kgp, Kep = Krauss_dictionary_U[f'Kgp{mu_p}'], Krauss_dictionary_U[f'Kep{mu_p}']
        probs_gp.append((Kgp.getH()@Kgp@rho).diagonal().sum())
        rho = Kgp @ rho @ Kgp.getH() + Kep @rho @ Kep.getH()
        mu_q = (mu_q +1)% 2
        rhos.append(rho.copy())
    
    return rhos, np.real(probs_gq), np.real(probs_gp)


def sBs_allbits(Delta,q0,p0,sensor,T,gauges):
    """
    R is the number of q-p cycles
    rho0 the initial state of the oscillator 
    gauges: starting gauge
    returns all bitstrings probabilities
    """
    beta = (q0 + 1j*p0)/np.sqrt(2)
    U_q0p0 = Displacement(beta)
    rho0 = U_q0p0@sensor@U_q0p0.getH()

    Krauss_dictionary, Krauss_dictionary_U = Krauss_dictionaries(Delta)
    rho = rho0.copy()
    rhos = [rho.copy()]
    probs_b = []
    
    def Krauss_from_b(b):
        mu_q, mu_p = gauges[0], gauges[1] #initialize the gauges
        Krauss_b = I.copy()
        bq, bp = b[::2], b[1::2]
        for j in range(len(bq)):
            KgqU, KeqU = Krauss_dictionary_U[f'Kgq{mu_q}'], Krauss_dictionary_U[f'Keq{mu_q}']
            if bq[j] == '0':
                Krauss_b = KgqU @ Krauss_b 
            else:
                Krauss_b = KeqU @ Krauss_b
            mu_p = (mu_p +1)% 2#gauge update

            KgpU, KepU = Krauss_dictionary_U[f'Kgp{mu_p}'], Krauss_dictionary_U[f'Kep{mu_p}']
            if bp[j] == '0':
                Krauss_b = KgpU @ Krauss_b
            else:
                Krauss_b = KepU @ Krauss_b
            mu_q = (mu_q +1)% 2#gauge update
            
        return Krauss_b


def sBs_allqbits(Delta,q0,p0,sensor,T,gauges):
    """
    R is the number of q-p cycles
    rho0 the initial state of the oscillator 
    gauges: starting gauge
    returns all q bitstrings probabilities. 
    """
    beta = (q0 + 1j*p0)/np.sqrt(2)
    U_q0p0 = Displacement(beta)
    rho0 = U_q0p0@sensor@U_q0p0.getH()

    Krauss_dictionary, Krauss_dictionary_U = Krauss_dictionaries(Delta)
    rho = rho0.copy()
    
    def Krauss_q_from_bq(bq):
        mu_q, mu_p = gauges[0], gauges[1] #initialize the gauges
        Krauss_b = I.copy()
        for j in range(len(bq)):
            KgqU, KeqU = Krauss_dictionary_U[f'Kgq{mu_q}'], Krauss_dictionary_U[f'Keq{mu_q}']
            if bq[j] == '0':
                Krauss_b = KgqU @ Krauss_b 
            else:
                Krauss_b = KeqU @ Krauss_b
            mu_p = (mu_p +1)% 2#gauge update


            KgpU, KepU = Krauss_dictionary_U[f'Kgp{mu_p}'], Krauss_dictionary_U[f'Kep{mu_p}']
            Krauss_b = (KgpU+KeqU) @ Krauss_b
            mu_q = (mu_q +1)% 2#gauge update
        return Krauss_b
    
    def prob_from_b(bq,rho):
        bq = int_to_r_bits(bq,T)
        mu_q, mu_p = gauges[0], gauges[1] #initialize the gauges
        for j in range(len(bq)):
            KgqU, KeqU = Krauss_dictionary_U[f'Kgq{mu_q}'], Krauss_dictionary_U[f'Keq{mu_q}']
            if bq[j] == '0':
                rho = KgqU @ rho @ KgqU.getH()
            else:
                rho = KeqU @ rho @ KeqU.getH()
            mu_p = (mu_p +1)% 2#gauge update

            KgpU, KepU = Krauss_dictionary_U[f'Kgp{mu_p}'], Krauss_dictionary_U[f'Kep{mu_p}']
            rho = KgpU @ rho @ KgpU.getH() + KepU @ rho @ KepU.getH()
            mu_q = (mu_q +1)% 2#gauge update
        return (rho).diagonal().sum()

    probs = Parallel(n_jobs=16)(delayed(prob_from_b)(bq,rho) for bq in range(2**(T)))
    # probs = []
    # for bq in range(2**T):
    #     probs.append(prob_from_b(bq,rho))
    return np.real(np.array(probs))

def sBs_allpbits(Delta,q0,p0,sensor,T,gauges):
    """
    R is the number of q-p cycles
    rho0 the initial state of the oscillator 
    gauges: starting gauge
    returns all p bitstrings probabilities. 
    """
    beta = (q0 + 1j*p0)/np.sqrt(2)
    U_q0p0 = Displacement(beta)
    rho0 = U_q0p0@sensor@U_q0p0.getH()

    Krauss_dictionary, Krauss_dictionary_U = Krauss_dictionaries(Delta)
    rho = rho0.copy()
    rhos = [rho.copy()]
    probs_b = []
    
    def Krauss_p_from_bp(bp):
        mu_q, mu_p = gauges[0], gauges[1] #initialize the gauges
        Krauss_b = I.copy()
        for j in range(len(bq)):
            KgqU, KeqU = Krauss_dictionary_U[f'Kgq{mu_q}'], Krauss_dictionary_U[f'Keq{mu_q}']
            Krauss_b = (KgqU+KeqU) @ Krauss_b
            mu_p = (mu_p +1)% 2#gauge update

            KgpU, KepU = Krauss_dictionary_U[f'Kgp{mu_p}'], Krauss_dictionary_U[f'Kep{mu_p}']
            if bp[j] == '0':
                Krauss_b = KgpU @ Krauss_b
            else:
                Krauss_b = KepU @ Krauss_b
            mu_q = (mu_q +1)% 2#gauge update
        return Krauss_b
    
    def prob_from_b(bp,rho):
        bp = int_to_r_bits(bp,T)
        Krauss = Krauss_p_from_bp(bp)
        return np.real((Krauss.getH()@Krauss@rho).diagonal().sum())

    probs = Parallel(n_jobs=16)(delayed(prob_from_b)(bp,rho) for bp in range(2**(T)))
    return [np.real(np.array(probs))]


def sBs_random(Delta,q0,p0,sensor,T, Krauss_dictionary, Krauss_dictionary_U):
    #R is the number of q-p cycles
    #rho0 the initial state of the oscillator   

    rho = rho0.copy()
    mu_q, mu_p = 0, 1 #initialize the gauges
    Kgq, Keq = Krauss_dictionary[f'Kgq{mu_q}'], Krauss_dictionary[f'Keq{mu_q}']
    Kgp, Kep = Krauss_dictionary[f'Kgp{mu_p}'], Krauss_dictionary[f'Kep{mu_p}']
    pq = (Kgq.getH()@Kgq@rho).diagonal().sum()
    pp = (Kgp.getH()@Kgp@rho).diagonal().sum()
    probs_gq, probs_gp = [], []
    for i in range(T):
        Kgq, Keq = Krauss_dictionary[f'Kgq{mu_q}'], Krauss_dictionary[f'Keq{mu_q}']
        Kgp, Kep = Krauss_dictionary[f'Kgp{mu_p}'], Krauss_dictionary[f'Kep{mu_p}']
        KgqU, KeqU = Krauss_dictionary_U[f'Kgq{mu_q}'], Krauss_dictionary_U[f'Keq{mu_q}']
        KgpU, KepU = Krauss_dictionary_U[f'Kgp{mu_p}'], Krauss_dictionary_U[f'Kep{mu_p}']
        mu_p = (mu_p +1)% 2#gauge update
        mu_q = (mu_q +1)% 2
        probs_gq.append((Kgq.getH()@Kgq@rho).diagonal().sum())
        #projective measurement of the qubit
        if np.random.rand() < probs_gq[-1]:
            
            rho = KgqU @ rho @ KgqU.getH() /probs_gq[-1]
        else:
            rho = KeqU @ rho @ KeqU.getH() /(1-probs_gq[-1])

        probs_gp.append((Kgp.getH()@Kgp@rho).diagonal().sum())
        #projective measurement of the qubit
        if np.random.rand() < probs_gp[-1]:
            rho = KgpU @ rho @ KgpU.getH() /probs_gp[-1]
        else: 
            rho = KepU @ rho @ KepU.getH() /(1-probs_gp[-1])
    probs_gq = np.array(probs_gq)
    probs_gp = np.array(probs_gp)
    probs = np.array([probs_gq, probs_gp])
    return probs

def sBs_random_run(Delta,rho,T, Krauss_dictionary, Krauss_dictionary_U,gauges):
    '''
    single run of sBs with measurement and feedback.
    '''
    mu_q, mu_p = gauges[0], gauges[1] #start gauges
    bits, probs_q, probs_p = [],[],[]
    for i in range(T):
        Kgq, Keq = Krauss_dictionary[f'Kgq{mu_q}'], Krauss_dictionary[f'Keq{mu_q}']
        KgqU, KeqU = Krauss_dictionary_U[f'Kgq{mu_q}'], Krauss_dictionary_U[f'Keq{mu_q}']
        pgq = np.real((Kgq.getH()@Kgq@rho).diagonal().sum())
        #projective measurement of the qubit
        if np.random.rand() < pgq:
            bits.append(0)
            rho = KgqU @ rho @ KgqU.getH()/(KgqU @ rho @ KgqU.getH()).diagonal().sum()
            probs_q.append(pgq)
        else:
            bits.append(1)
            rho = KeqU @ rho @ KeqU.getH() /(KeqU @ rho @ KeqU.getH()).diagonal().sum()
            probs_q.append(1-pgq)
        mu_p = (mu_p +1)% 2#gauge update

        Kgp, Kep = Krauss_dictionary[f'Kgp{mu_p}'], Krauss_dictionary[f'Kep{mu_p}']
        KgpU, KepU = Krauss_dictionary_U[f'Kgp{mu_p}'], Krauss_dictionary_U[f'Kep{mu_p}']
        pgp = np.real((Kgp.getH()@Kgp@rho).diagonal().sum())
        #projective measurement of the qubit
        if np.random.rand() < pgp:
            bits.append(0)
            rho = KgpU @ rho @ KgpU.getH() /(KgpU @ rho @ KgpU.getH()).diagonal().sum()
            probs_p.append(pgp)
        else: 
            bits.append(1)
            rho = KepU @ rho @ KepU.getH() /(KepU @ rho @ KepU.getH()).diagonal().sum()
            probs_p.append(1-pgp)
        mu_q = (mu_q +1)% 2
    final_gauge = [mu_q,mu_p]
    return rho, np.array(bits), np.array(probs_q), np.array(probs_p),final_gauge