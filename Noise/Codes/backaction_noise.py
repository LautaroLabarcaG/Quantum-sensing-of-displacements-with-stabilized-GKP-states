import qutip as qu
import numpy as np
import numpy.random as random
from typing import Dict, Tuple, Callable, List, Union, Optional
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml
import importlib
import general_functions
import bits
importlib.reload(general_functions)
importlib.reload(bits)
from general_functions import *
from general_functions import Displacement
from metrology_noise import *
from bits import *


def sBs_random_bits(rhoc, system_dict: Dict): #to be written yet
    """
    Evolution of the sensor state with rounds of stabilisation (with sBs) with random bitstring    
    Input:
    - rhoc: density matrix of the cavity state
    - system_dict : dictionary containing all the parameters defining the system.
    Return:
    - rhof : final state of the system
    - bitstring: bitstring used for the evolution
    """
    g_proj = qu.tensor(
        qu.basis(system_dict["dimensions"]["qubit"], 1)
        * qu.basis(system_dict["dimensions"]["qubit"], 1).dag(),
        qu.qeye(system_dict["dimensions"]["cavity"]),
    )
    e_proj = qu.tensor(
        qu.basis(system_dict["dimensions"]["qubit"], 0)
        * qu.basis(system_dict["dimensions"]["qubit"], 0).dag(),
        qu.qeye(system_dict["dimensions"]["cavity"]),
    )

    # definition of rho_plus = |+><+|
    psi_plus = (
        qu.basis(system_dict["dimensions"]["qubit"], 0)
        + qu.basis(system_dict["dimensions"]["qubit"], 1)
    ) / np.sqrt(2)
    rho_plus = psi_plus * psi_plus.dag()

    proj_list = [g_proj, e_proj]

    # Number of rounds
    T = system_dict["simulations"]["T"]

    bit_arr = []

    q1 = np.arange(0, 2 * T, 2)
    p1 = np.arange(1, 2 * T + 1, 2) 
    rho = qu.tensor(rho_plus, rhoc)
    gauge_q = 0
    gauge_p = 0

    for round in range(T):
        # q round
        rho, pgq = sBs_q_evolve(rho, system_dict, gauge=gauge_q)
        if np.real(pgq)>random.random():
            bit_arr.append(0)
            rho = proj_list[0] * rho
        else:
            bit_arr.append(1)
            rho = proj_list[1] * rho
        rho = qu.tensor(rho_plus, rho.ptrace(1))
        rho = rho/rho.tr()
        gauge_p = (gauge_p + 1) % 2
        # p round
        rho, pgp = sBs_p_evolve(rho, system_dict, gauge=gauge_p)
        if np.real(pgp)>random.random():
            bit_arr.append(0)
            rho = proj_list[0] * rho
        else:
            bit_arr.append(1)
            rho = proj_list[1] * rho
        rho = qu.tensor(rho_plus, rho.ptrace(1))
        rho = rho/rho.tr()
        gauge_q = (gauge_q + 1) % 2
    
    # Final state
    rhof = rho.ptrace(1)
    return rhof, bit_arr

def sBs_random_bits_probs_estimators(q0:float, p0:float, rho_init, system_dict:Dict, estimators_q_p:List):
    """
    Evolution of the sensor state with rounds of stabilisation (with sBs) with random bitstring    
    Input:
    - q0: initial displacement
    - p0: initial displacement
    - system_dict : dictionary containing all the parameters defining the system. See backaction_evading function for details.
    - rho_init: initial state.
    - estimators_expectation_dict: dictionary with
        - estimators_allqbits_p0: array with the estimators of q0 and p0 for each bitstring of length T. Shape [2**(T)]. Could be obtained from ML or Bayesian.
        - estimators_allpbits_q0: array with the estimators of q0 and p0 for each bitstring of length T. Shape [2**(T)]. Could be obtained from ML or Bayesian.
        - exp_q_allqbits: expected value of q after a displacement q0 in qrange, p0=l/4, and recovery bitstring b. Shape [][].
        - exp_p_allpbits: expected value of p after a displacement p0 in prange, q0=l/4, and recovery bitstring b. Shape [][].
    Return:
    - rhof : final state of the system
    - bitstring: bitstring used for the evolution
    """

    # Displacement
    alpha = (q0 + 1j * p0) / np.sqrt(2)
    a = system_dict["operators"]["cavity"]["a"]
    U_alpha =  (alpha * a.dag() - np.conj(alpha) * a).expm()
    # print(U_alpha.shape)
    # print(rho_init.shape)
    rhoc = U_alpha * rho_init * U_alpha.dag()
    
    rhof, bitstring = sBs_random_bits(rhoc, system_dict)
    # Estimators
    bq, bp = bitstring[::2], bitstring[1::2]
    bq = binary_array_to_int(bq)
    bp = binary_array_to_int(bp)
    # print('bitstring',bitstring)
    # print('bq',bq)
    # print('bp',bp)

    estimators_allqbits_p0 = estimators_q_p[0]
    estimators_allpbits_q0 = estimators_q_p[1]
    estimatorq = estimators_allqbits_p0[bq]
    estimatorp = estimators_allpbits_q0[bp]

    return rhof, bitstring, estimatorq, estimatorp



def sBs_autonomous(rho_init, system_dict: Dict):
    """
    Evolution of the sensor state with rounds of sBs stabilisation without measurement.
    This is done to stabilize the state and bring it back to the center of phase-space in the backaction evading protocol. 
    Input:
    - v : Displacement value
    - system_dict : dictionary containing all the parameters defining the system.
    Return:
    - rho_f : final state after stabilization
    """
    g_proj = qu.tensor(
        qu.basis(system_dict["dimensions"]["qubit"], 1)
        * qu.basis(system_dict["dimensions"]["qubit"], 1).dag(),
        qu.qeye(system_dict["dimensions"]["cavity"]),
    )
    e_proj = qu.tensor(
        qu.basis(system_dict["dimensions"]["qubit"], 0)
        * qu.basis(system_dict["dimensions"]["qubit"], 0).dag(),
        qu.qeye(system_dict["dimensions"]["cavity"]),
    )

    # definition of rho_plus = |+><+|
    psi_plus = (
        qu.basis(system_dict["dimensions"]["qubit"], 1)
        + qu.basis(system_dict["dimensions"]["qubit"], 0)
    ) / np.sqrt(2)
    rho_plus = psi_plus * psi_plus.dag()

    proj_list = [g_proj, e_proj]

    # Number of rounds
    M = system_dict["simulations"]["M"] #note the addition of M in compared with previous sBs functions.     
    rho = qu.tensor(rho_plus, rho_init)

    gauge_q = 0
    gauge_p = 0
    # print('inital gauge',[gauge_q,gauge_p])

    for idx_ in range(M):
        rho, pgq = sBs_q_evolve(rho, system_dict, gauge=gauge_q)
        rhof = rho #autonomous
        rho = qu.tensor(rho_plus, rhof.ptrace(1))
        # gauge update
        gauge_p = (gauge_p + 1) % 2
        # print('after q','Tq',np.round((T_qdelta * rhof.ptrace(1)).tr(),3),'Tp',np.round((T_pdelta * rhof.ptrace(1)).tr(),3),'v',np.round(v,6),'gauge',[gauge_q,gauge_p])
 

        # p round
        rho, pgp = sBs_p_evolve(rho, system_dict, gauge=gauge_p)
        rhof = rho #autonomous
        rho = qu.tensor(rho_plus, rhof.ptrace(1))
        # gauge update
        gauge_q = (gauge_q + 1) % 2
        # print('after p','Tq',np.round((T_qdelta * rhof.ptrace(1)).tr(),3),'Tp',np.round((T_pdelta * rhof.ptrace(1)).tr(),3),'v',np.round(v,6),'gauge',[gauge_q,gauge_p])

    # print('final Tq',(T_qdelta * rhof.ptrace(1)).tr(),'final Tp',(T_pdelta * rhof.ptrace(1)).tr())
    rhof = rho.ptrace(1)
    # print('trace',rhof.tr())
    return rhof

def backaction_evading(system_dict:Dict):
    '''
        q0: initial displacement
        p0: initial displacement
        system_dict: dictionary with the system parameters. These include:
            - dimensions
                - qubit: number of levels of the qubit
                - cavity: number of levels of the cavity
            - operators
                - qubit
                    - sx, sy, sz: pauli operators
                    - sm: qubit lowering operator
                - cavity
                    - a: cavity lowering operator
            - state_params
                - delta: envelope variance
                - l: length of lattice
            - timings
                - t_B: times for the big displacement simulation
            - c_ops: list with the collapse operators, these include the level of noise
            - simulations
                - decay_ratio: the coefficient of the decay
                - T: number of q-p cycles for estimation (even)
                - M: number of autonomous restabilization rounds (even)
                - N: number of repetitions of the protocol, same convention as in the paper
                - idx_data_q, idx_data_p: index of the data to be used. From the data get estimators
                - v_lim: range of the q0 displacements in the data
                - n_v: number of steps in the range of q0 displacements in the data
                - stddev: standard deviation of the initial displacement
                - sensor: initial state of the sensor
            - data
                - idx_data_q: index of the q quadrature data. Stored in qorppath
                - idx_data_p: index of the p quadrature data. Stored in qorppath
                - path_data: path to the data, qororppath     
    '''
    # print('backaction evading with rhos')

    N_c = system_dict["dimensions"]["cavity"]
    delta = system_dict["state_params"]["delta"]
    rho_init = system_dict["simulations"]["sensor"]
    a = system_dict["operators"]["cavity"]["a"]

    squared_errors_q, squared_errors_p = [], []
    estimators_q, estimators_p = [], []
    q0s, p0s = [], []

    #load the data and build the estimators and final expected values dictionary
    name1 = 'Data - qbits - params'+str(system_dict["data"]["idx_data_q"])
    results1 = np.load(system_dict["data"]["path_data"]+name1+'.npz')
    vlist1 = results1['vlist']
    p_array1 = results1['p_array']
    qf_array1 = results1['qf_array']
    pf_array1 = results1['pf_array']

    # print('shape', rho_init.shape)
    # print('purity initial',(qu.Qobj(rho_init)**2).tr())

    name2 = 'Data - qbits - params'+str(system_dict["data"]["idx_data_p"])
    results2 = np.load(system_dict["data"]["path_data"]+name2+'.npz')
    vlist2 = results2['vlist']
    p_array2 = results2['p_array']
    qf_array2 = results2['qf_array']
    pf_array2 = results2['pf_array']

    l = np.sqrt(2 * np.pi)
    stddev = system_dict["simulations"]["stddev"] * l # should be 0.15l or 0.10l
    estimators_allqbits_p0 = estimators_from_parray(p_array1,vlist1,stddev) #estimator associated to each bitstring of qbits, for fix p0. Not ideal, but close to the ideal.
    estimators_allpbits_q0 = estimators_from_parray(p_array2,vlist2,stddev) #estimator associated to each bitstring of pbits, for fix q0

    estimators_expectation_dict = {}
    estimators_expectation_dict["estimators_allqbits_p0"] = estimators_allqbits_p0
    estimators_expectation_dict["estimators_allpbits_q0"] = estimators_allpbits_q0
    estimators_expectation_dict["exp_q_allqbits"] = qf_array1
    estimators_expectation_dict["exp_p_allpbits"] = pf_array2

    #start the lists with errors, estimators, and density matrices
    squared_errors_q, squared_errors_p = [], []
    estimators_q, estimators_p = [], []
    rhos = []
    q0s, p0s = [], []
    rho = rho_init.copy()
    for i in range(system_dict["simulations"]["N"]):
        q0 = np.random.normal(0,stddev)
        p0 = np.random.normal(0,stddev)
        q0s.append(q0)
        p0s.append(p0)

        rho, bitstring, estq, estp, estq_final, estp_final = sBs_random_bits_probs_estimators(q0,p0,rho,system_dict,estimators_expectation_dict)
        estimators_q.append(estq)
        estimators_p.append(estp) 
        squared_errors_q.append((estq-q0)**2)
        squared_errors_p.append((estp-p0)**2)

        alpha = -(estq_final+1j*estp_final)/2
        U_alpha = (alpha * a.dag() - np.conj(alpha) * a).expm()
        rho = U_alpha * rho * U_alpha.dag()
        #now, restabilize the state
        rho = sBs_autonomous(rho, system_dict)
        rhos.append(rho) #choose small N, just to check the density matrices are not diverging and all is ok.
    
    # print('purity final',(qu.Qobj(rhos[-1])**2).tr())


    squared_errors_q = np.array(squared_errors_q)
    squared_errors_p = np.array(squared_errors_p)
    estimators_q = np.array(estimators_q)
    estimators_p = np.array(estimators_p)
    q0s = np.array(q0s)
    p0s = np.array(p0s)
    return [squared_errors_q, squared_errors_p, estimators_q, estimators_p, q0s, p0s, rhos]


def backaction_evading_notrhos(system_dict:Dict):
    '''
        q0: initial displacement
        p0: initial displacement
        system_dict: dictionary with the system parameters. These include:
            - dimensions
                - qubit: number of levels of the qubit
                - cavity: number of levels of the cavity
            - operators
                - qubit
                    - sx, sy, sz: pauli operators
                    - sm: qubit lowering operator
                - cavity
                    - a: cavity lowering operator
            - state_params
                - delta: envelope variance
                - l: length of lattice
            - timings
                - t_B: times for the big displacement simulation
            - c_ops: list with the collapse operators, these include the level of noise
            - simulations
                - decay_ratio: the coefficient of the decay
                - T: number of q-p cycles for estimation (even)
                - M: number of autonomous restabilization rounds (even)
                - N: number of repetitions of the protocol, same convention as in the paper
                - idx_data_q, idx_data_p: index of the data to be used. From the data get estimators
                - v_lim: range of the q0 displacements in the data
                - n_v: number of steps in the range of q0 displacements in the data
                - stddev: standard deviation of the initial displacement
                - sensor: initial state of the sensor
            - data
                - idx_data_q: index of the q quadrature data. Stored in qorppath
                - idx_data_p: index of the p quadrature data. Stored in qorppath
                - path_data: path to the data, qororppath     
    '''
    # print('backaction evading no rhos')

    N_c = system_dict["dimensions"]["cavity"]
    delta = system_dict["state_params"]["delta"]
    rho_init = system_dict["simulations"]["sensor"]
    a = system_dict["operators"]["cavity"]["a"]

    squared_errors_q, squared_errors_p = [], []
    estimators_q, estimators_p = [], []
    q0s, p0s = [], []

    #load the data and build the estimators and final expected values dictionary
    name1 = 'Data - qbits - params'+str(system_dict["data"]["idx_data_q"])
    results1 = np.load(system_dict["data"]["path_data"]+name1+'.npz')
    vlist1 = results1['vlist']
    p_array1 = results1['p_array']
    qf_array1 = results1['qf_array']
    pf_array1 = results1['pf_array']

    # print('shape', rho_init.shape)
    # print('purity initial',(qu.Qobj(rho_init)**2).tr())

    name2 = 'Data - qbits - params'+str(system_dict["data"]["idx_data_p"])
    results2 = np.load(system_dict["data"]["path_data"]+name2+'.npz')
    vlist2 = results2['vlist']
    p_array2 = results2['p_array']
    qf_array2 = results2['qf_array']
    pf_array2 = results2['pf_array']

    l = np.sqrt(2 * np.pi)
    stddev = system_dict["simulations"]["stddev"] * l # should be 0.15l or 0.10l 
    estimators_allqbits_p0 = estimators_from_parray(p_array1,vlist1,stddev) #estimator associated to each bitstring of qbits, for fix p0. Not ideal, but close to the ideal.
    estimators_allpbits_q0 = estimators_from_parray(p_array2,vlist2,stddev) #estimator associated to each bitstring of pbits, for fix q0

    estimators_expectation_dict = {}
    estimators_expectation_dict["estimators_allqbits_p0"] = estimators_allqbits_p0
    estimators_expectation_dict["estimators_allpbits_q0"] = estimators_allpbits_q0
    estimators_expectation_dict["exp_q_allqbits"] = qf_array1
    estimators_expectation_dict["exp_p_allpbits"] = pf_array2

    #start the lists with errors, estimators, and density matrices
    squared_errors_q, squared_errors_p = [], []
    estimators_q, estimators_p = [], []
    bitstrings = []
    q0s, p0s = [], []
    rho = rho_init.copy()
    for i in range(system_dict["simulations"]["N"]):
        q0 = np.random.normal(0,stddev)
        p0 = np.random.normal(0,stddev)
        q0s.append(q0)
        p0s.append(p0)

        rho, bitstring, estq, estp, estq_final, estp_final = sBs_random_bits_probs_estimators(q0,p0,rho,system_dict,estimators_expectation_dict)
        estimators_q.append(estq)
        estimators_p.append(estp) 
        squared_errors_q.append((estq-q0)**2)
        squared_errors_p.append((estp-p0)**2)
        bitstrings.append(bitstring)

        alpha = -(estq_final+1j*estp_final)/2
        U_alpha = (alpha * a.dag() - np.conj(alpha) * a).expm()
        rho = U_alpha * rho * U_alpha.dag()
        #now, restabilize the state
        rho = sBs_autonomous(rho, system_dict)
    
    # print('purity final',(qu.Qobj(rhos[-1])**2).tr())

    squared_errors_q = np.array(squared_errors_q)
    squared_errors_p = np.array(squared_errors_p)
    estimators_q = np.array(estimators_q)
    estimators_p = np.array(estimators_p)
    q0s = np.array(q0s)
    p0s = np.array(p0s)
    return [squared_errors_q, squared_errors_p, estimators_q, estimators_p, q0s, p0s, bitstrings]

def backaction_evading_sBs_fidelities_probs_allqbits_simplerecovery(q0:float, system_dict:Dict):
    """
    q0: initial displacement, assume p0 = 0.
    system_dict: dictionary with the system parameters. These include:
        - dimensions
            - qubit: number of levels of the qubit
            - cavity: number of levels of the cavity
        - operators
            - qubit
                - sx, sy, sz: pauli operators
                - sm: qubit lowering operator
            - cavity
                - a: cavity lowering operator
        - state_params
            - delta: envelope variance
            - l: length of lattice
        - timings
            - t_B: times for the big displacement simulation
        - c_ops: list with the collapse operators, these include the level of noise
        - simulations
            - decay_ratio: the coefficient of the decay
            - T: number of q-p cycles for estimation (even)
            - M: number of autonomous restabilization rounds (even)
            - N: number of repetitions of the protocol, same convention as in the paper
            - idx_data_q, idx_data_p: index of the data to be used. From the data get estimators
            - v_lim: range of the q0 displacements in the data
            - n_v: number of steps in the range of q0 displacements in the data
            - stddev: standard deviation of the initial displacement
            - sensor: initial state of the sensor
        - data
            - idx_data_q: index of the q quadrature data. Stored in qorppath
            - idx_data_p: index of the p quadrature data. Stored in qorppath
            - path_data: path to the data, qororppath     

    Returns the fidelities with the initial state for all bitstrings of the qbits, after the backaction evading protocol with sBs.
    Also returns the probability of each bitstring
    """
    
    alpha = q0/np.sqrt(2)
    a = system_dict["operators"]["cavity"]["a"]
    U_alpha =  (alpha * a.dag() - np.conj(alpha) * a).expm()
    rho_init = system_dict["simulations"]["sensor"]
    rhoc = U_alpha * rho_init * U_alpha.dag() # state after the displacement

    #load the data and build the estimators dictionary
    name1 = 'Data - qbits - params'+str(system_dict["data"]["idx_data_q"])
    # name1 = 'Data - qbits - params'+str(151)
    results1 = np.load(system_dict["data"]["path_data"]+name1+'.npz')
    vlist1 = results1['vlist']
    p_array1 = results1['p_array']
    qf_array1 = results1['qf_array']
    pf_array1 = results1['pf_array']

    p_array1_T = []
    for i in range(len(vlist1)):
        p_array1_T.append(marginalize_probs(p_array1[i], system_dict["simulations"]["T"])) #marginalize the probabilities to get the probabilities of each bitstring of qbits, for a given displacement q0
    p_array1_T = np.array(p_array1_T)
    stddev = system_dict["simulations"]["stddev"] * l # should be 0.15l or 0.10l 
    estimators_allqbits_p0 = estimators_from_parray(p_array1_T,vlist1,stddev) #estimator associated to each bitstring of qbits, for fix p0. Not ideal, but close to the ideal.
    
    #First get the probabilities of each bitstring, and state after the bitstring evolution.
    #use the estimator of that bistring to displace the state. 
    #run M rounds of autonomous sBs stabilisation
    #compute the fidelity with the initial state.
    #for each bitstring, we will have the fidelity and the probability.

    g_proj = qu.tensor(
            qu.basis(system_dict["dimensions"]["qubit"], 1)
            * qu.basis(system_dict["dimensions"]["qubit"], 1).dag(),
            qu.qeye(system_dict["dimensions"]["cavity"]),
        )
    e_proj = qu.tensor(
        qu.basis(system_dict["dimensions"]["qubit"], 0)
        * qu.basis(system_dict["dimensions"]["qubit"], 0).dag(),
        qu.qeye(system_dict["dimensions"]["cavity"]),
    )

    # definition of rho_plus = |+><+|
    psi_plus = (
        qu.basis(system_dict["dimensions"]["qubit"], 0)
        + qu.basis(system_dict["dimensions"]["qubit"], 1)
    ) / np.sqrt(2)
    rho_plus = psi_plus * psi_plus.dag()

    proj_list = [g_proj, e_proj]

    T = system_dict["simulations"]["T"]
    M = system_dict["simulations"]["M"]

    bitstring_arr = np.array(list(itertools.product([0, 1], repeat= T))) 
    p_arr = np.zeros(len(bitstring_arr))
    fidelity_arr = np.zeros(len(bitstring_arr))
    traces_arr = np.zeros(len(bitstring_arr))
    purity_arr = np.zeros(len(bitstring_arr))

    for idx_, y_ in enumerate(bitstring_arr): #T bits of sBs metrology
        q1 = np.arange(0, T, 1)
        rho = qu.tensor(rho_plus, rhoc.copy())
        gauge_q = 0
        gauge_p = 0
        for idx_T, round in enumerate(np.linspace(0, T - 1, T)):
            # q round
            rho, pgq = sBs_q_evolve(rho, system_dict, gauge=gauge_q)
            rhof = proj_list[bitstring_arr[idx_, q1[idx_T]]] * rho
            rho = qu.tensor(rho_plus, rhof.ptrace(1))
            # gauge update
            gauge_p = (gauge_p + 1) % 2

            # p round
            rho, pgp = sBs_p_evolve(rho, system_dict, gauge=gauge_p)
            rhof = rho #autonomous
            rho = qu.tensor(rho_plus, rhof.ptrace(1))
            # gauge update
            gauge_q = (gauge_q + 1) % 2

        rho = rho.ptrace(1)# final state after the bitstring evolution, note this last ptrace does not change the state. 
        p_arr[idx_] = np.real(rhof.tr()) #probability of the bitstring
        rho = rho / rho.tr()  # normalize the state
        if system_dict['recovery'] == 'yes':
            estimatorq = estimators_allqbits_p0[idx_]
            alpha = -estimatorq / np.sqrt(2)
            U_alpha = (alpha * a.dag() - np.conj(alpha) * a).expm()
            rho = U_alpha * rho * U_alpha.dag() #displace the state
        rho = qu.tensor(rho_plus, rho) #rebuild the state with the qubit in the |+> state

        for idx_M in range(M):# M rounds of autonomous sBs stabilisation
            rho, pgq = sBs_q_evolve(rho, system_dict, gauge=gauge_q)
            rhof = rho #autonomous
            rho = qu.tensor(rho_plus, rhof.ptrace(1))
            # gauge update
            gauge_p = (gauge_p + 1) % 2
    
            # p round
            rho, pgp = sBs_p_evolve(rho, system_dict, gauge=gauge_p)
            rhof = rho #autonomous
            rho = qu.tensor(rho_plus, rhof.ptrace(1))
            # gauge update
            gauge_q = (gauge_q + 1) % 2
        rhof = rho.ptrace(1)
        rhof = rhof / rhof.tr()  # normalize the state

        traces_arr[idx_] = np.real(rhof.tr()) #should be 1
        purity_arr[idx_] = np.real((rhof**2).tr())
        fidelity_arr[idx_] = qt.fidelity(rhof, rho_init)
    return [fidelity_arr, p_arr, bitstring_arr, traces_arr, purity_arr]

def backaction_evading_sBs_fidelities_probs_allqbits_simplerecovery_parallel(system_dict:Dict, params):
    '''
    runs the backaction evading protocol with sBs for all bitstrings of the qbits, in parallel.

    Returns the fidelities with the initial state for all bitstrings of the qbits, after the backaction evading protocol with sBs.
    Also returns the probability of each bitstring. 
    '''
    l = np.sqrt(2 * np.pi)
    qlist = np.linspace(
        0, 
        params["simulations"]["v_lim"] * l,#0.3
        params["simulations"]["n_v"], # steps 20
    )
    
    parallel = Parallel(n_jobs=params["simulations"]["n_jobs"])
    results = parallel(
        delayed(backaction_evading_sBs_fidelities_probs_allqbits_simplerecovery)(q0, system_dict) for q0 in qlist
    )
    fidelities = np.array([result[0] for result in results])
    probabilities = np.array([result[1] for result in results])
    bitstrings = np.array([result[2] for result in results])
    traces = np.array([result[3] for result in results])
    purities = np.array([result[4] for result in results])
    return fidelities, probabilities, bitstrings, traces, purities



def backaction_evading_sBs_simplerecovery(system_dict:Dict):
    """
    q0: initial displacement, assume p0 = 0.
    system_dict: dictionary with the system parameters. These include:
        - dimensions
            - qubit: number of levels of the qubit
            - cavity: number of levels of the cavity
        - operators
            - qubit
                - sx, sy, sz: pauli operators
                - sm: qubit lowering operator
            - cavity
                - a: cavity lowering operator
        - state_params
            - delta: envelope variance
            - l: length of lattice
        - timings
            - t_B: times for the big displacement simulation
        - c_ops: list with the collapse operators, these include the level of noise
        - simulations
            - decay_ratio: the coefficient of the decay
            - T: number of q-p cycles for estimation (even)
            - M: number of autonomous restabilization rounds (even)
            - N: number of repetitions of the protocol, same convention as in the paper
            - reps: number of repetitions of the protocol, same convention as in the paper
            - idx_data_q, idx_data_p: index of the data to be used. From the data get estimators
            - v_lim: range of the q0 displacements in the data
            - n_v: number of steps in the range of q0 displacements in the data
            - stddev: standard deviation of the initial displacement
            - sensor: initial state of the sensor
        - data
            - idx_data_q: index of the q quadrature data. Stored in qorppath
            - idx_data_p: index of the p quadrature data. Stored in qorppath
            - path_data: path to the data, qororppath     

    Returns the fidelities with the initial state for all bitstrings of the qbits, after the backaction evading protocol with sBs.
    Also returns the probability of each bitstring
    """
    
    a = system_dict["operators"]["cavity"]["a"]
    rho_init = system_dict["simulations"]["sensor"]

    #load the data and build the estimators dictionary
    name1 = 'Data - qbits - params'+str(system_dict["data"]["idx_data_q"])
    # name1 = 'Data - qbits - params'+str(151)
    results1 = np.load(system_dict["data"]["path_data"]+name1+'.npz')
    vlist1 = results1['vlist']
    p_array1 = results1['p_array']
    qf_array1 = results1['qf_array']
    pf_array1 = results1['pf_array']

    name2 = 'Data - qbits - params'+str(system_dict["data"]["idx_data_p"])
    results2 = np.load(system_dict["data"]["path_data"]+name2+'.npz')
    vlist2 = results2['vlist']
    p_array2 = results2['p_array']
    qf_array2 = results2['qf_array']
    pf_array2 = results2['pf_array']

    p_array1_T = []
    p_array2_T = []
    for i in range(len(vlist1)):
        p_array1_T.append(marginalize_probs(p_array1[i], system_dict["simulations"]["T"])) #marginalize the probabilities to get the probabilities of each bitstring of qbits, for a given displacement q0
        p_array2_T.append(marginalize_probs(p_array2[i], system_dict["simulations"]["T"])) #marginalize the probabilities to get the probabilities of each bitstring of pbits, for a given displacement p0
    p_array1_T = np.array(p_array1_T)
    p_array2_T = np.array(p_array2_T)
    stddev = system_dict["simulations"]["stddev"] * l # should be 0.15l or 0.10l 
    estimators_allqbits_p0 = estimators_from_parray(p_array1_T,vlist1,stddev) #estimator associated to each bitstring of qbits, for fix p0. Not ideal, but close to the ideal.
    estimators_allpbits_q0 = estimators_from_parray(p_array2_T,vlist2,stddev) #estimator associated to each bitstring of pbits, for fix q0
    estimators_q_p = [estimators_allqbits_p0, estimators_allpbits_q0]
    #start the lists with errors, estimators
    squared_errors_q, squared_errors_p = [], []
    estimators_q, estimators_p = [], []
    bitstrings = []
    q0s, p0s = [], []
    rho = rho_init.copy()
    for i in range(system_dict["simulations"]["N"]):
        q0 = np.random.normal(0,stddev)
        p0 = np.random.normal(0,stddev)
        q0s.append(q0)
        p0s.append(p0)

        rho, bitstring, estq, estp = sBs_random_bits_probs_estimators(q0,p0,rho,system_dict,estimators_q_p)
        estimators_q.append(estq)
        estimators_p.append(estp) 
        squared_errors_q.append((estq-q0)**2)
        squared_errors_p.append((estp-p0)**2)
        bitstrings.append(bitstring)

        alpha = -(estq+1j*estp)/2
        U_alpha = (alpha * a.dag() - np.conj(alpha) * a).expm()
        rho = U_alpha * rho * U_alpha.dag()
        #now, restabilize the state
        rho = sBs_autonomous(rho, system_dict)
    
    # print('purity final',(qu.Qobj(rhos[-1])**2).tr())

    squared_errors_q = np.array(squared_errors_q)
    squared_errors_p = np.array(squared_errors_p)
    estimators_q = np.array(estimators_q)
    estimators_p = np.array(estimators_p)
    q0s = np.array(q0s)
    p0s = np.array(p0s)
    return [squared_errors_q, squared_errors_p, estimators_q, estimators_p, q0s, p0s, bitstrings]





def backaction_evading_sBs_simplerecovery_parallel(system_dict:Dict, params):
    '''
    runs the backaction evading protocol with sBs for all bitstrings of the qbits, in parallel.

    Returns the fidelities with the initial state for all bitstrings of the qbits, after the backaction evading protocol with sBs.
    Also returns the probability of each bitstring. 
    '''
    l = np.sqrt(2 * np.pi)
    parallel = Parallel(n_jobs=params["simulations"]["n_jobs"])
    results = parallel(
        delayed(backaction_evading_sBs_simplerecovery)(system_dict) for i in tqdm(range(system_dict["simulations"]["reps"])))
    squared_errors_q = np.array([result[0] for result in results])
    squared_errors_p = np.array([result[1] for result in results])
    estimators_q = np.array([result[2] for result in results])
    estimators_p = np.array([result[3] for result in results])
    q0s = np.array([result[4] for result in results])
    p0s = np.array([result[5] for result in results])
    bitstrings = np.array([result[6] for result in results])
    return squared_errors_q, squared_errors_p, estimators_q, estimators_p, q0s, p0s, bitstrings

    
