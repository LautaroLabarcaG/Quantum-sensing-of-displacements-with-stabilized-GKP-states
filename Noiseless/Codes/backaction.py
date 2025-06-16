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
from sBs import *
from metrology import *

def sBs_cycle_returns_gauge(Delta,R,rho0,gauges):
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
    
    final_gauge = [mu_q,mu_p]
    return rhos, np.real(probs_gq), np.real(probs_gp), final_gauge

def sBs_cycle_returns_gauge_lastrho(Delta,R,rho0,gauges):
    #R is the number of q-p cycles
    #rho0 the initial state of the oscillator 
    #returns all probabilities. Length of the list is R+1.

    Krauss_dictionary, Krauss_dictionary_U = Krauss_dictionaries(Delta)
    rho = rho0.copy()
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
    
    final_gauge = [mu_q,mu_p]
    return rho, np.real(probs_gq), np.real(probs_gp), final_gauge

def backaction_evading_sBs_run(Delta,q0,p0,sensor,repeat_baem, T,M,estimators_allqbits_p0,estimators_allpbits_q0,gauges,exp_qrange_allqbits):
    '''
        Delta:envelope variance
        q0,p0: initial displacement
        sensor: initial state
        repeat_baem: number of repetitions of the protocol (N)
        T: number of rounds of q-p cycles for estimation 
        M: number of rounds of q-p cycles for restabilization 
        estimators_allqbits_q0p0: array with the estimators of q0 and p0 for each bitstring of length T. Shape [2**(T)]. Could be obtained from ML or Bayesian. 
        gauges: array with the initial gauges.
        exp_qrange_allqbits: expected value of q after a displacement q0 in qrange, p0=l/4, and recovery bitstring b. Shape [q0][2**T].

        runs the backaction evading sBs protocol.

        Returns, rho, squared_errors_q, squared_errors_p, estimators_q, estimators_p
    '''
    rho = sensor.copy()
    squared_errors_q, squared_errors_p = [], []
    estimators_q, estimators_p = [], []
    rhos = []
    steps = 101
    qrange = (l)*np.linspace(-1.0,1.0,steps)#range of the exp_qrange_allqbits

    for i in range(repeat_baem):
        #first, the bits used for estimation 
        rho,final_gauge, bitstring, estq, estp, estq_final, estp_final = sBs_random_bits_probs_estimators(Delta,q0,p0,rho,T,estimators_allqbits_p0,estimators_allpbits_q0,gauges,exp_qrange_allqbits,qrange,steps)
        estimators_q.append(estq)
        estimators_p.append(estp) 
        squared_errors_q.append((estq-q0)**2)
        squared_errors_p.append((estp-p0)**2)

        alpha = -(estq_final+1j*estp_final)/2
        U_alpha = Displacement(alpha)
        rho = U_alpha @ rho @ U_alpha.getH()
        #now, restabilize the state
        rho, probs_gq, probs_gp, final_gauge = sBs_cycle_returns_gauge_lastrho(Delta,M,rho,gauges)
        rhos.append(rho)
    
    squared_errors_q = np.array(squared_errors_q)
    squared_errors_p = np.array(squared_errors_p)
    estimators_q = np.array(estimators_q)
    estimators_p = np.array(estimators_p)
    return rhos, squared_errors_q, squared_errors_p, estimators_q, estimators_p
    

def backaction_evading_sBs_run_notrhos(Delta,sigma,sensor,repeat_baem, T,M,estimators_allqbits_p0,estimators_allpbits_q0,gauges,exp_qrange_allqbits):
    '''
        Delta:envelope variance
        q0,p0: initial displacement
        sensor: initial state
        repeat_baem: number of repetitions of the protocol (N)
        T: number of rounds of q-p cycles for estimation 
        M: number of rounds of q-p cycles for restabilization 
        estimators_allqbits_q0p0: array with the estimators of q0 and p0 for each bitstring of length T. Shape [2**(T)]. Could be obtained from ML or Bayesian. 
        gauges: array with the initial gauges.
        exp_qrange_allqbits: expected value of q after a displacement q0 in qrange, p0=l/4, and recovery bitstring b. Shape [q0][2**T].

        runs the backaction evading sBs protocol.

        Returns, squared_errors_q, squared_errors_p, estimators_q, estimators_p
    '''
    rho = sensor.copy()
    squared_errors_q, squared_errors_p = [], []
    estimators_q, estimators_p = [], []
    q0s, p0s = [], []
    steps = 101
    qrange = (l)*np.linspace(-1.0,1.0,steps)#range of the exp_qrange_allqbits
    bitstrings = []

    for i in range(repeat_baem):
        q0 = np.random.normal(0,sigma)
        p0 = np.random.normal(0,sigma)
        q0s.append(q0)
        p0s.append(p0)
        rho,gauges, bitstring, estq, estp, estq_final, estp_final = sBs_random_bits_probs_estimators(Delta,q0,p0,rho,T,estimators_allqbits_p0,estimators_allpbits_q0,gauges,exp_qrange_allqbits,qrange,steps)
        estimators_q.append(estq)
        estimators_p.append(estp)
        squared_errors_q.append((estq-q0)**2)
        squared_errors_p.append((estp-p0)**2)
        bitstrings.append(bitstring)

        alpha = -(estq_final+1j*estp_final)/2
        U_alpha = Displacement(alpha)
        rho = U_alpha @ rho @ U_alpha.getH()
        #now, restabilize the state
        rho, probs_gq, probs_gp, gauges = sBs_cycle_returns_gauge_lastrho(Delta,M,rho,gauges)
    
    squared_errors_q = np.array(squared_errors_q)
    squared_errors_p = np.array(squared_errors_p)
    estimators_q = np.array(estimators_q)
    estimators_p = np.array(estimators_p)
    q0s = np.array(q0s)
    p0s = np.array(p0s)
    return squared_errors_q, squared_errors_p, estimators_q, estimators_p, q0s, p0s, bitstrings

def backaction_evading_sBs_run_notrhos_simple(Delta,sigma,sensor,repeat_baem, T,M,estimators_allqbits_p0,estimators_allpbits_q0,gauges,exp_qrange_allqbits):
    '''
        Delta:envelope variance
        q0,p0: initial displacement
        sensor: initial state
        repeat_baem: number of repetitions of the protocol (N)
        T: number of rounds of q-p cycles for estimation 
        M: number of rounds of q-p cycles for restabilization 
        estimators_allqbits_q0p0: array with the estimators of q0 and p0 for each bitstring of length T. Shape [2**(T)]. Could be obtained from ML or Bayesian. 
        gauges: array with the initial gauges.
        exp_qrange_allqbits: expected value of q after a displacement q0 in qrange, p0=l/4, and recovery bitstring b. Shape [q0][2**T].

        runs the backaction evading sBs protocol.

        Returns, squared_errors_q, squared_errors_p, estimators_q, estimators_p
    '''
    rho = sensor.copy()
    squared_errors_q, squared_errors_p = [], []
    estimators_q, estimators_p = [], []
    q0s, p0s = [], []
    steps = 101
    qrange = (l)*np.linspace(-1.0,1.0,steps)#range of the exp_qrange_allqbits
    bitstrings = []

    for i in range(repeat_baem):
        q0 = np.random.normal(0,sigma)
        p0 = np.random.normal(0,sigma)
        q0s.append(q0)
        p0s.append(p0)
        rho,gauges, bitstring, estq, estp, estq_final, estp_final = sBs_random_bits_probs_estimators(Delta,q0,p0,rho,T,estimators_allqbits_p0,estimators_allpbits_q0,gauges,exp_qrange_allqbits,qrange,steps)
        estimators_q.append(estq)
        estimators_p.append(estp)
        squared_errors_q.append((estq-q0)**2)
        squared_errors_p.append((estp-p0)**2)
        bitstrings.append(bitstring)

        alpha = -(estq+1j*estp)/2
        U_alpha = Displacement(alpha)
        rho = U_alpha @ rho @ U_alpha.getH()
        #now, restabilize the state
        rho, probs_gq, probs_gp, gauges = sBs_cycle_returns_gauge_lastrho(Delta,M,rho,gauges)
    
    squared_errors_q = np.array(squared_errors_q)
    squared_errors_p = np.array(squared_errors_p)
    estimators_q = np.array(estimators_q)
    estimators_p = np.array(estimators_p)
    q0s = np.array(q0s)
    p0s = np.array(p0s)
    return squared_errors_q, squared_errors_p, estimators_q, estimators_p, q0s, p0s, bitstrings


def backaction_evading_sBs_fidelities_probs_allqbits(Delta,q0,p0,sensor,T,M,estimators_allqbits_p0,estimators_allpbits_q0,gauges,exp_qrange_allqbits, sensor_gauge):
    '''
        Delta:envelope variance
        q0,p0: initial displacement
        sensor: initial state
        repeat_baem: number of repetitions of the protocol (N)
        T: number of rounds of q-p cycles for estimation 
        M: number of rounds of q-p cycles for restabilization 
        estimators_allqbits_q0p0: array with the estimators of q0 and p0 for each bitstring of length T. Shape [2**(T)]. Could be obtained from ML or Bayesian. 
        gauges: array with the initial gauges.
        exp_qrange_allqbits: expected value of q after a displacement q0 in qrange, p0=l/4, and recovery bitstring b. Shape [q0][2**T].

        runs the backaction evading sBs protocol.

        Returns the fidelity of the final state with the initial state for each possible bitstring.
    '''
    rho = sensor.copy()
    qrange = (l)*np.linspace(-1.0,1.0,101) #range of the exp_qrange_allqbits

    beta = (q0 + 1j*p0)/np.sqrt(2)
    U_q0p0 = Displacement(beta)
    rho0 = U_q0p0 @ rho @ U_q0p0.getH()

    Krauss_dictionary, Krauss_dictionary_U = Krauss_dictionaries(Delta)
    rho = rho0.copy()

    def fidelity_from_bq(bq,rho):
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
        final_gauge = [mu_q,mu_p]
        p_b = (rho).diagonal().sum()

        #now apply the recovery displacement
        bq = binary_array_to_int(bq)
        estimatorq = estimators_allqbits_p0[bq]
        index_q0 = np.argmin(np.abs(qrange-estimatorq))
        estq_final = exp_qrange_allqbits[index_q0][bq]
        alpha = -estq_final/np.sqrt(2)
        U_alpha = Displacement(alpha)
        rho = U_alpha @ rho @ U_alpha.getH()

        rho = rho/(rho).diagonal().sum()

        #now apply the restabilization
        rho, probs_gq, probs_gp, final_gauge = sBs_cycle_returns_gauge_lastrho(Delta,M,rho,final_gauge)
        return [qt.fidelity(qt.Qobj(rho),qt.Qobj(sensor_gauge)),p_b]

    fidelities_probs_allqbits = Parallel(n_jobs=16)(delayed(fidelity_from_bq)(bq,rho) for bq in range(2**(T)))
    return np.array(fidelities_probs_allqbits)

def backaction_evading_sBs_fidelities_probs_allqbits_simplerecovery(Delta,q0,p0,sensor,T,M,estimators_allqbits_p0,estimators_allpbits_q0,gauges,exp_qrange_allqbits, sensor_gauge):
    '''
        Delta:envelope variance
        q0,p0: initial displacement
        sensor: initial state
        repeat_baem: number of repetitions of the protocol (N)
        T: number of rounds of q-p cycles for estimation 
        M: number of rounds of q-p cycles for restabilization 
        estimators_allqbits_q0p0: array with the estimators of q0 and p0 for each bitstring of length T. Shape [2**(T)]. Could be obtained from ML or Bayesian. 
        gauges: array with the initial gauges.
        exp_qrange_allqbits: expected value of q after a displacement q0 in qrange, p0=l/4, and recovery bitstring b. Shape [q0][2**T].

        runs the backaction evading sBs protocol.

        Returns the fidelity of the final state with the initial state for each possible bitstring.
    '''
    rho = sensor.copy()
    qrange = (l)*np.linspace(-1.0,1.0,101) #range of the exp_qrange_allqbits

    beta = (q0 + 1j*p0)/np.sqrt(2)
    U_q0p0 = Displacement(beta)
    rho0 = U_q0p0 @ rho @ U_q0p0.getH()

    Krauss_dictionary, Krauss_dictionary_U = Krauss_dictionaries(Delta)
    rho = rho0.copy()

    def fidelity_from_bq(bq,rho):
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
        final_gauge = [mu_q,mu_p]
        p_b = (rho).diagonal().sum()

        #now apply the recovery displacement
        bq = binary_array_to_int(bq)
        estimatorq = estimators_allqbits_p0[bq]
        # index_q0 = np.argmin(np.abs(qrange-estimatorq))
        # estq_final = exp_qrange_allqbits[index_q0][bq]
        alpha = -estimatorq/np.sqrt(2)
        U_alpha = Displacement(alpha)
        rho = U_alpha @ rho @ U_alpha.getH()

        rho = rho/(rho).diagonal().sum()

        #now apply the restabilization
        rho, probs_gq, probs_gp, final_gauge = sBs_cycle_returns_gauge_lastrho(Delta,M,rho,final_gauge)
        purity = (qt.Qobj(rho) * qt.Qobj(rho)).tr()
        return [qt.fidelity(qt.Qobj(rho),qt.Qobj(sensor_gauge)),p_b, purity]

    fidelities_probs_purity_allqbits = Parallel(n_jobs=16)(delayed(fidelity_from_bq)(bq,rho) for bq in range(2**(T)))
    return np.array(fidelities_probs_purity_allqbits)


def backaction_evading_sBs_fidelities_probs_allqbits_filterrecovery(Delta,q0,p0,sensor,T,M,estimators_allqbits_p0,estimators_allpbits_q0,gauges,exp_qrange_allqbits, sensor_gauge):
    '''
        Delta:envelope variance
        q0,p0: initial displacement
        sensor: initial state
        repeat_baem: number of repetitions of the protocol (N)
        T: number of rounds of q-p cycles for estimation 
        M: number of rounds of q-p cycles for restabilization 
        estimators_allqbits_q0p0: array with the estimators of q0 and p0 for each bitstring of length T. Shape [2**(T)]. Could be obtained from ML or Bayesian. 
        gauges: array with the initial gauges.
        exp_qrange_allqbits: expected value of q after a displacement q0 in qrange, p0=l/4, and recovery bitstring b. Shape [q0][2**T].

        runs the backaction evading sBs protocol.

        Returns the fidelity of the final state with the initial state for each possible bitstring.
    '''
    rho = sensor.copy()
    qrange = (l)*np.linspace(-1.0,1.0,101) #range of the exp_qrange_allqbits

    beta = (q0 + 1j*p0)/np.sqrt(2)
    U_q0p0 = Displacement(beta)
    rho0 = U_q0p0 @ rho @ U_q0p0.getH()

    Krauss_dictionary, Krauss_dictionary_U = Krauss_dictionaries(Delta)
    rho = rho0.copy()
    def filter(estimatorq):
        # Apply the sinh filter to the estimator
        return np.tanh((12-T) * l * estimatorq)**(12+T)

    def filter(estimatorq, T, Delta): #this would be actual function written in the main, it also works, it was just harder to write a fit code as curve_fit doesn't take it.
            b, c = 1.44,-.44
            qlist = [estimatorq]
            for T in np.arange(1,T+1):
                qt = qlist[T-1].copy()
                rate2 = b+c*np.abs(np.sin(l*qt))
                qt = qt*np.exp(-rate2*Delta**2*T)
                qlist.append(qt)
            
            qfinal = qlist[-1]
            c1 = 0.03 + 0 * 0.003 * T
            c2 = 0.005
            a = np.arctanh(1/1.1)/c1/l
            b = np.arctanh(1.1 * np.tanh(a * c2 * l))/c2/l
            f2 = 1 - np.tanh(b * estimatorq)
            f1 = 1.1 * np.tanh(a * estimatorq) * np.tanh(50 * estimatorq)**10
            # return f1 * estimatorq + f2 * qfinal
            if estimatorq > 0.08 * l:
                return estimatorq
            else:
                return qfinal

    def fidelity_from_bq(bq,rho):
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
        final_gauge = [mu_q,mu_p]
        p_b = (rho).diagonal().sum()

        #now apply the recovery displacement
        bq = binary_array_to_int(bq)
        estimatorq = estimators_allqbits_p0[bq]
        # index_q0 = np.argmin(np.abs(qrange-estimatorq))
        # estq_final = exp_qrange_allqbits[index_q0][bq]
        
        alpha = -filter(estimatorq, T, Delta)/np.sqrt(2)
        U_alpha = Displacement(alpha)
        rho = U_alpha @ rho @ U_alpha.getH()

        rho = rho/(rho).diagonal().sum()

        #now apply the restabilization
        rho, probs_gq, probs_gp, final_gauge = sBs_cycle_returns_gauge_lastrho(Delta,M,rho,final_gauge)
        return [qt.fidelity(qt.Qobj(rho),qt.Qobj(sensor_gauge)),p_b]

    fidelities_probs_allqbits = Parallel(n_jobs=16)(delayed(fidelity_from_bq)(bq,rho) for bq in range(2**(T)))
    return np.array(fidelities_probs_allqbits)

def backaction_evading_sBs_fidelities_probs_allqbits_norecovery(Delta,q0,p0,sensor,T,M,estimators_allqbits_p0,estimators_allpbits_q0,gauges,exp_qrange_allqbits, sensor_gauge):
    '''
        Delta:envelope variance
        q0,p0: initial displacement
        sensor: initial state
        repeat_baem: number of repetitions of the protocol (N)
        T: number of rounds of q-p cycles for estimation 
        M: number of rounds of q-p cycles for restabilization 
        estimators_allqbits_q0p0: array with the estimators of q0 and p0 for each bitstring of length T. Shape [2**(T)]. Could be obtained from ML or Bayesian. 
        gauges: array with the initial gauges.
        exp_qrange_allqbits: expected value of q after a displacement q0 in qrange, p0=l/4, and recovery bitstring b. Shape [q0][2**T].

        runs the backaction evading sBs protocol.

        Returns the fidelity of the final state with the initial state for each possible bitstring.
    '''
    rho = sensor.copy()

    beta = (q0 + 1j*p0)/np.sqrt(2)
    U_q0p0 = Displacement(beta)
    rho0 = U_q0p0 @ rho @ U_q0p0.getH()

    Krauss_dictionary, Krauss_dictionary_U = Krauss_dictionaries(Delta)
    rho = rho0.copy()

    def fidelity_from_bq(bq,rho):
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

        p_b = (rho).diagonal().sum()
        rho = rho/(rho).diagonal().sum()

        #now apply the restabilization
        final_gauge = [mu_q,mu_p]
        rho, probs_gq, probs_gp, final_gauge = sBs_cycle_returns_gauge_lastrho(Delta,M,rho,final_gauge)
        return [qt.fidelity(qt.Qobj(rho),qt.Qobj(sensor_gauge)),p_b]

    fidelities_probs_allqbits = Parallel(n_jobs=16)(delayed(fidelity_from_bq)(bq,rho) for bq in range(2**(T)))
    return np.array(fidelities_probs_allqbits)


def backaction_mixed_rho(sigma,Delta,R,sensor,gauges,steps,p0):
    '''
    sigma: sampling standard deviation
    Delta: envelope
    R: number of rounds of q-p cycles
    sensor: initial state ideal sensor state
    gauges: initial gauges
    steps: steps for the q0range

    returns: the mixed density matrix after displacement an R rounds of stabilization 
    '''

    q0range = (l)*np.linspace(-1.0,1.0,steps)#range of the exp_qrange_allqbits
    prior = np.random.normal(0,sigma,steps)
    p0 = 0
    rhos = []
    rhos = Parallel(n_jobs=16)(delayed(sBs_cycle_finalrho)(Delta,R,sensor,gauges,q0,p0) for q0 in tqdm(q0range))
    # for q0 in tqdm(q0range):
    #     rho = sensor.copy()
    #     beta = (q0 + 1j*p0)/np.sqrt(2)
    #     U_q0p0 = Displacement(beta)
    #     rho0 = U_q0p0 @ rho @ U_q0p0.getH()

    #     rhos.append(sBs_cycle_finalrho(Delta,R,rho0,gauges))
    
    mix_rho = 0*sensor
    for idx in range(steps):
        mix_rho += prior[idx]*rhos[idx]
    mix_rho = mix_rho/(mix_rho).diagonal().sum()
    return mix_rho