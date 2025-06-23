import qutip as qu
import numpy as np
from typing import Dict, Tuple, Callable, List, Union, Optional
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml

# print('new')

def use0Fonts(fontSize):
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = "cmr10"
    mpl.rcParams["text.usetex"] = True
    font = {"size": fontSize}
    mpl.rc("font", **font)

def Displacement(alpha: float = 0, system_dict: Dict = {"operators": qu.destroy(140)}):
    """
    Definition of the displacement operator.
    Input:
    - alpha : displacement amplitude
    - system_dict : dictionary containing all the parameters defining the system. This dict. should have a subdictionary with operators of the system, in this case the destruction operator of the cavity mode.
    Return:
    - Displacement operator in the form exp(alpha * a.dag() - conj(alpha) * a)
    """
    a = system_dict["operators"]["cavity"]["a"]
    return (alpha * a.dag() - np.conj(alpha) * a).expm()


def R_x(theta: float = 0, system_dict: Dict = {}):
    """
    Definition of Rotation operator around x axis.
    Input:
    - theta : rotation angle
    - system_dict : dictionary containing all the parameters defining the system. This dict. should have a subdictionary with operator sigma_x and subdictionary of the system dimensions.
    Return:
    - Rotation operator in the form exp(i * theta * sigma_x / 2)
    """
    sx = system_dict["operators"]["qubit"]["sx"]
    N_c = system_dict["dimensions"]["cavity"]
    return qu.tensor((-1.0j * theta * sx / 2), qu.qeye(N_c)).expm()


def CD(alpha: float = 0, system_dict: Dict = {}):
    """
    Definition of the conditionnal displacement operator.
    Input:
    - alpha : conditional displacement amplitude
    - system_dict : dictionary containing all the parameters defining the system. This dict. should have a subdictionary with operators of the system, in this case the destruction operator of the cavity mode and sigma_z of the qubit mode.
    Return:
    - Conditionnal displacement operator in the form exp(alpha * a.dag() - conj(alpha) * a) x sigma_z
    """
    a = system_dict["operators"]["cavity"]["a"]
    sz = system_dict["operators"]["qubit"]["sz"]
    return (
        qu.tensor(sz, (alpha * a.dag() - np.conj(alpha) * a)) / (2 * np.sqrt(2))
    ).expm()


def H_CD(alpha: float = 0, system_dict: Dict = {}):
    """
    Definition of the conditionnal displacement hamiltonian used for the time evolution of big displacement in sBs.
    Input:
    - alpha : conditional displacement amplitude
    - system_dict : dictionary containing all the parameters defining the system. This dict. should have a subdictionary with operators of the system, in this case the destruction operator of the cavity mode and sigma_z of the qubit mode.
    Return:
    - Conditional displacement hamiltonian
    """
    a = system_dict["operators"]["cavity"]["a"]
    sz = system_dict["operators"]["qubit"]["sz"]
    # kerr = system_dict["state_params"]["kerr"]
    return (1.0j / (2 * np.sqrt(2))) * qu.tensor(
        sz, (alpha * a.dag() - np.conj(alpha) * a)
    )

def H_CD_kerr(alpha: float = 0, system_dict: Dict = {}):
    """
    Definition of the conditionnal displacement hamiltonian used for the time evolution of big displacement in sBs.
    Input:
    - alpha : conditional displacement amplitude
    - system_dict : dictionary containing all the parameters defining the system. This dict. should have a subdictionary with operators of the system, in this case the destruction operator (and kerr operator) of the cavity mode and sigma_z of the qubit mode.
    Return:
    - Conditional displacement hamiltonian
    """
    a = system_dict["operators"]["cavity"]["a"]
    sz = system_dict["operators"]["qubit"]["sz"]
    kerr_op = system_dict["operators"]["cavity"]["kerr"]#adag @ adag @ a @ a
    # kerr = system_dict["state_params"]["kerr"]
    return (1.0j / (2 * np.sqrt(2))) * qu.tensor(
        sz, (alpha * a.dag() - np.conj(alpha) * a)
    )

def sBs_q_evolve(rho: qu.Qobj, system_dict: Dict = {}, gauge: int = 0):
    """
    Definition of the sBs sequence for stabilization along q axis.
    Input:
    - rho : "initial" density matrix
    - system_dict : dictionary containing all the parameters defining the system.
    Return:
    - rho : final density matrix after sBs round
    - p_g : probability of finding the qubit in ground after sBs round
    """

    ## Definition of parameters
    t_B = system_dict["timings"]["t_B"]
    #
    N_q = system_dict["dimensions"]["qubit"]
    N_c = system_dict["dimensions"]["cavity"]
    #
    delta = system_dict["state_params"]["delta"]
    l = system_dict["state_params"]["l"]
    #
    c_ops = system_dict["c_ops"]
    g_proj = qu.tensor(qu.basis(N_q, 1) * qu.basis(N_q, 1).dag(), qu.qeye(N_c))
    coshd = np.cosh(delta ** 2)

    ## Definition of sBs sequence
    rho = (
        CD(np.sinh(delta ** 2) * l / 2, system_dict)
        * rho
        * CD(np.sinh(delta ** 2) * l / 2, system_dict).dag()
    )
    rho = R_x(np.pi / 2, system_dict) * rho * R_x(np.pi / 2, system_dict).dag()
    rho = qu.mesolve(
        H_CD(-1.0j * l * np.cosh(delta**2), system_dict), rho, t_B, system_dict["c_ops"], []
    ).states[-1]
    rho_i = (
        R_x(((-1) ** gauge) * np.pi / 2, system_dict).dag()
        * rho
        * R_x(((-1) ** gauge) * np.pi / 2, system_dict)
    )
    rho = (
        CD(np.sinh(delta ** 2) * l / 2, system_dict)
        * rho_i
        * CD(np.sinh(delta ** 2) * l / 2, system_dict).dag()
    )
    #
    p_g = (rho * g_proj).tr()

    return rho, p_g


def sBs_p_evolve(rho: qu.Qobj, system_dict: Dict = {}, gauge: int = 0):
    """
    Definition of the sBs sequence for stabilization along p axis.
    Input:
    - rho : "initial" density matrix
    - system_dict : dictionary containing all the parameters defining the system.
    Return:
    - rho : final density matrix after sBs round
    - p_g : probability of finding the qubit in ground after sBs round
    """

    ## Definition of parameters
    t_B = system_dict["timings"]["t_B"]
    #
    N_q = system_dict["dimensions"]["qubit"]
    N_c = system_dict["dimensions"]["cavity"]
    #
    delta = system_dict["state_params"]["delta"]
    l = system_dict["state_params"]["l"]
    #
    c_ops = system_dict["c_ops"]
    g_proj = qu.tensor(qu.basis(N_q, 1) * qu.basis(N_q, 1).dag(), qu.qeye(N_c))
    coshd = np.cosh(delta**2)

    ## Definition of sBs sequence
    rho = (
        CD(1.0j * np.sinh(delta ** 2) * l / 2, system_dict)
        * rho
        * CD(1.0j * np.sinh(delta ** 2) * l / 2, system_dict).dag()
    )
    rho = R_x(np.pi / 2, system_dict) * rho * R_x(np.pi / 2, system_dict).dag()
    rho = qu.mesolve(H_CD(1.0 * l * np.cosh(delta**2), system_dict), rho, t_B, system_dict["c_ops"], []).states[
        -1
    ]
    rho_i = (
        R_x(((-1) ** gauge) * np.pi / 2, system_dict).dag()
        * rho
        * R_x(((-1) ** gauge) * np.pi / 2, system_dict)
    )
    rho = (
        CD(1.0j * np.sinh(delta ** 2) * l / 2, system_dict)
        * rho_i
        * CD(1.0j * np.sinh(delta ** 2) * l / 2, system_dict).dag()
    )
    #
    p_g = (rho * g_proj).tr()

    return rho, p_g


def p_gaussian_prior(v: list, var: float = 1):
    """
    Definition of the gaussian prior probability distribution
    Input:
    - v : list of displacement
    - variance : variance of the gaussian prior
    Return:
    - probability distribution of a gaussian prior
    """
    p_v = (1.0 / (np.sqrt(2 * np.pi) * var)) * np.exp(-(1 / (2 * var)) * v ** 2)
    return p_v / sum(p_v)


def prob_evolution(p_arr_v, r):
    """
    Definition of the function to look at the evolution of probabilities as function of the number of rounds.
    Input:
    - p_arr_v : list of probabilities as function of rounds for a given displacement v
    - r : number of rounds 
    Return:
    - p_arr_r : probability list as function of rounds
    """
    len_bitstring = 2 ** (2 * r)
    cut_indices = np.ones((r))
    for idx, cut_i in enumerate(cut_indices):
        cut_indices[idx] = len_bitstring / (2 ** (2 * idx + 1))
    p_arr_r = np.zeros((r))
    for idx, cut_i in enumerate(cut_indices):
        cut_i = int(cut_i)
        p = 0
        for idx_b, p_b in enumerate(p_arr_v):
            if ((idx_b) // cut_i) % 2 == 0:
                p += p_b
        p_arr_r[idx] = p
    return p_arr_r


def MSD_v(p_array: List, vlist: List, variance: float = 1, system_dict: Dict = {}):
    """
    Definition of Mean Square Deviation for a given variance.
    Input:
    - vlist : list of displacement
    - variance : variance of the gaussian prior
    - system_dict : dictionary containing all the parameters defining the system.
    Return:
    - MSD : probability distribution of a gaussian prior
    - posterior_arr : posterior distribution
    """

    def p_gaussian_prior(v: List, variance: float = 1):
        """
        Definition of the gaussian prior probability distribution
        Input:
        - v : list of displacement
        - variance : variance of the gaussian prior
        Return:
        - probability distribution of a gaussian prior
        """
        p_v = (1.0 / (np.sqrt(2 * np.pi) * variance)) * np.exp(
            -(1 / (2 * variance)) * v ** 2
        )
        return p_v / sum(p_v)
    
    def p_flat_prior(v:List, variance: float=1):
        """
        Definition of the flat prior probability distribution
        Input:
        - v : list of displacement
        - variance : variance of the flat prior
        Return:
        - probability distribution of a flat prior
        """
        p_v = np.ones(len(v))
        return p_v / sum(p_v)

    # Definition of parameters
    ds = abs(vlist[1] - vlist[0])
    r = system_dict["simulations"]["r"]

    estimator_arr = np.empty((2 ** (2 * r)))
    posterior_arr = np.empty((2 ** (2 * r), len(vlist)))

    for idx in range(2 ** (2 * r)):
        # probability of bitstring y
        p_y = sum(p_array[:, idx] * p_gaussian_prior(vlist, variance)) * ds #gaussian prior
        # p_y = sum(p_array[:, idx] * p_flat_prior(vlist, variance)) * ds #flat prior
        # posterior distribution
        posterior = p_array[:, idx] * p_gaussian_prior(vlist, variance) / p_y #gaussian posterior
        # posterior = p_array[:, idx] * p_flat_prior(vlist, variance) / p_y #flat posterior

        posterior_arr[idx, :] = posterior
        # estimator
        estimator_arr[idx] = ds * sum(vlist * posterior)

    MSD_v_ = np.sum(
        p_array
        * (
            np.ones((len(vlist), 2 ** (2 * r))) * estimator_arr
            - (vlist * np.ones((2 ** (2 * r), len(vlist)))).T
        )
        ** 2,
        axis=1,
    )
    MSD = p_gaussian_prior(vlist, variance) * np.array(MSD_v_)

    return MSD, MSD_v_, posterior_arr

def MSD_v_onlyqorp(p_array: List, vlist: List, variance: float = 1, system_dict: Dict = {}):
    """
    Definition of Mean Square Deviation for a given variance.
    Input:
    - vlist : list of displacement
    - variance : variance of the gaussian prior
    - system_dict : dictionary containing all the parameters defining the system.
    Return:
    - MSD : probability distribution of a gaussian prior
    - posterior_arr : posterior distribution
    """

    def p_gaussian_prior(v: List, variance: float = 1):
        """
        Definition of the gaussian prior probability distribution
        Input:
        - v : list of displacement
        - variance : variance of the gaussian prior
        Return:
        - probability distribution of a gaussian prior
        """
        p_v = (1.0 / (np.sqrt(2 * np.pi) * variance)) * np.exp(
            -(1 / (2 * variance)) * v ** 2
        )
        return p_v / sum(p_v)
    
    def p_flat_prior(v:List, variance: float=1):
        """
        Definition of the flat prior probability distribution
        Input:
        - v : list of displacement
        - variance : variance of the flat prior
        Return:
        - probability distribution of a flat prior
        """
        p_v = np.ones(len(v))
        return p_v / sum(p_v)

    # Definition of parameters
    ds = abs(vlist[1] - vlist[0])
    r = system_dict["simulations"]["r"]

    estimator_arr = np.empty((2 ** (r)))
    posterior_arr = np.empty((2 ** (r), len(vlist)))

    for idx in range(2 ** (r)):
        # probability of bitstring y
        p_y = sum(p_array[:, idx] * p_gaussian_prior(vlist, variance)) * ds #gaussian prior
        # p_y = sum(p_array[:, idx] * p_flat_prior(vlist, variance)) * ds #flat prior
        # posterior distribution
        posterior = p_array[:, idx] * p_gaussian_prior(vlist, variance) / p_y #gaussian posterior
        # posterior = p_array[:, idx] * p_flat_prior(vlist, variance) / p_y #flat posterior

        posterior_arr[idx, :] = posterior
        # estimator
        estimator_arr[idx] = ds * sum(vlist * posterior)

    MSD_v_ = np.sum(
        p_array
        * (
            np.ones((len(vlist), 2 ** (r))) * estimator_arr
            - (vlist * np.ones((2 ** (r), len(vlist)))).T
        )
        ** 2,
        axis=1,
    )
    MSD = p_gaussian_prior(vlist, variance) * np.array(MSD_v_)

    return MSD, MSD_v_, posterior_arr



def sBs_stabilization(v: float, system_dict: Dict):
    """
    Evolution of the sensor state with rounds of stabilisation (with sBs) for different bitstring possible
    Input:
    - v : Displacement value
    - system_dict : dictionary containing all the parameters defining the system.
    Return:
    - p_arr_r : probability list as function of rounds
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

    # Initial sensor state
    rho_init = system_dict["simulations"]["sensor"]

    # Number of rounds
    r = system_dict["simulations"]["r"]
    l = system_dict["state_params"]["l"]

    bitstring_arr = np.array(list(itertools.product([0, 1], repeat= 2 * r))) 
    p_arr = np.zeros(len(bitstring_arr))

    for idx_, y_ in enumerate(bitstring_arr):
        q1 = np.arange(0, 2 * r, 2)
        p1 = np.arange(1, 2 * r + 1, 2) 
        beta = (v + 1j * system_dict['simulations']['p_0'] * l) / np.sqrt(2)
        displaced_rho_0 = (
            Displacement(beta, system_dict)
            * rho_init
            * Displacement(beta, system_dict).dag()
        )
        rho = qu.tensor(rho_plus, displaced_rho_0)

        gauge_q = 0
        gauge_p = 0

        for idx_r, round in enumerate(np.linspace(0, r - 1, r)):
            # q round
            rho, pgq = sBs_q_evolve(rho, system_dict, gauge=gauge_q)
            rhof = proj_list[bitstring_arr[idx_, q1[idx_r]]] * rho
            rho = qu.tensor(rho_plus, rhof.ptrace(1))
            # gauge update
            gauge_p = (gauge_p + 1) % 2

            # p round
            rho, pgp = sBs_p_evolve(rho, system_dict, gauge=gauge_p)
            rhof = proj_list[bitstring_arr[idx_, p1[idx_r]]] * rho
            rho = qu.tensor(rho_plus, rhof.ptrace(1))
            # gauge update
            gauge_q = (gauge_q + 1) % 2

        p_arr[idx_] = np.real(rhof.tr())
    
    rhof = 0 # to avoid collapsing the memory
    return rhof, p_arr

def sBs_stabilization_q(v: float, system_dict: Dict):
    """
    Evolution of the sensor state with rounds of stabilisation (with sBs) for different bitstring possible
    Input:
    - v : Displacement value
    - system_dict : dictionary containing all the parameters defining the system.
    Return:
    - p_arr_r : probability list as function of rounds
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

    # Initial sensor state
    rho_init = system_dict["simulations"]["sensor"]

    # Number of rounds
    r = system_dict["simulations"]["r"]
    l = system_dict["state_params"]["l"]

    bitstring_arr = np.array(list(itertools.product([0, 1], repeat= 2 * r))) 
    p_arr = np.zeros(len(bitstring_arr))

    for idx_, y_ in enumerate(bitstring_arr):
        q1 = np.arange(0, 2 * r, 2)
        p1 = np.arange(1, 2 * r + 1, 2) 
        beta = (v + 1j * system_dict['simulations']['p_0'] * l) / np.sqrt(2)
        displaced_rho_0 = (
            Displacement(beta, system_dict)
            * rho_init
            * Displacement(beta, system_dict).dag()
        )
        rho = qu.tensor(rho_plus, displaced_rho_0)

        gauge_q = 0
        gauge_p = 0

        for idx_r, round in enumerate(np.linspace(0, r - 1, r)):
            # q round
            rho, pgq = sBs_q_evolve(rho, system_dict, gauge=gauge_q)
            rhof = proj_list[bitstring_arr[idx_, q1[idx_r]]] * rho
            rho = qu.tensor(rho_plus, rhof.ptrace(1))
            # gauge update
            gauge_p = (gauge_p + 1) % 2

            # p round
            rho, pgp = sBs_p_evolve(rho, system_dict, gauge=gauge_p)
            rhof = proj_list[bitstring_arr[idx_, p1[idx_r]]] * rho
            rho = qu.tensor(rho_plus, rhof.ptrace(1))
            # gauge update
            gauge_q = (gauge_q + 1) % 2

        p_arr[idx_] = np.real(rhof.tr())
    
    rhof = 0 # to avoid collapsing the memory
    return rhof, p_arr


def sBs_stabilization_p(v: float, system_dict: Dict):
    """
    Evolution of the sensor state with rounds of stabilisation (with sBs) for different bitstring possible
    Input:
    - v : Displacement value
    - system_dict : dictionary containing all the parameters defining the system.
    Return:
    - p_arr_r : probability list as function of rounds
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

    # Initial sensor state
    rho_init = system_dict["simulations"]["sensor"]

    # Number of rounds
    r = system_dict["simulations"]["r"]
    l = system_dict["state_params"]["l"]

    bitstring_arr = np.array(list(itertools.product([0, 1], repeat= 2 * r))) 
    p_arr = np.zeros(len(bitstring_arr))

    for idx_, y_ in enumerate(bitstring_arr):
        q1 = np.arange(0, 2 * r, 2)
        p1 = np.arange(1, 2 * r + 1, 2) 
        beta = (system_dict['simulations']['p_0'] * l + 1j * v) / np.sqrt(2)
        displaced_rho_0 = (
            Displacement(beta, system_dict)
            * rho_init
            * Displacement(beta, system_dict).dag()
        )
        rho = qu.tensor(rho_plus, displaced_rho_0)

        gauge_q = 0
        gauge_p = 0

        for idx_r, round in enumerate(np.linspace(0, r - 1, r)):
            # q round
            rho, pgq = sBs_q_evolve(rho, system_dict, gauge=gauge_q)
            rhof = proj_list[bitstring_arr[idx_, q1[idx_r]]] * rho
            rho = qu.tensor(rho_plus, rhof.ptrace(1))
            # gauge update
            gauge_p = (gauge_p + 1) % 2

            # p round
            rho, pgp = sBs_p_evolve(rho, system_dict, gauge=gauge_p)
            rhof = proj_list[bitstring_arr[idx_, p1[idx_r]]] * rho
            rho = qu.tensor(rho_plus, rhof.ptrace(1))
            # gauge update
            gauge_q = (gauge_q + 1) % 2

        p_arr[idx_] = np.real(rhof.tr())
    
    rhof = 0 # to avoid collapsing the memory
    return rhof, p_arr

def sBs_stabilization_qbitspath(v: float, system_dict: Dict):
    """
    Evolution of the sensor state with rounds of stabilisation (with sBs) for different q-bitstring possible. That is, the p quadrature is autonomous, and the q is measured.
    This is done to build the estimators up to 8 bits per quadrature. 
    Input:
    - v : Displacement value
    - system_dict : dictionary containing all the parameters defining the system.
    Return:
    - p_arr_r : probability list as function of rounds
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

    # Initial sensor state
    rho_init = system_dict["simulations"]["sensor"]

    # Number of rounds
    r = system_dict["simulations"]["r"]
    l = system_dict["state_params"]["l"]

    bitstring_arr = np.array(list(itertools.product([0, 1], repeat= r))) #r instead of 2r
    p_arr = np.zeros(len(bitstring_arr))
    qf_arr, pf_arr = np.zeros(len(bitstring_arr)), np.zeros(len(bitstring_arr))

    displaced_rho_0 = (
            Displacement((v + 1.j * system_dict['simulations']['p_0'] * l) / np.sqrt(2), system_dict)
            * rho_init
            * Displacement((v + 1.j * system_dict['simulations']['p_0'] * l) / np.sqrt(2), system_dict).dag()
        )


    for idx_, y_ in enumerate(bitstring_arr):
        q1 = np.arange(0, r, 1)
        # p1 = np.arange(1, 2 * r + 1, 2) #no needed for autonomous
        rho = qu.tensor(rho_plus, displaced_rho_0)
        gauge_q = 0
        gauge_p = 0

        for idx_r, round in enumerate(np.linspace(0, r - 1, r)):
            # q round
            rho, pgq = sBs_q_evolve(rho, system_dict, gauge=gauge_q)
            rhof = proj_list[bitstring_arr[idx_, q1[idx_r]]] * rho
            rho = qu.tensor(rho_plus, rhof.ptrace(1))
            # gauge update
            gauge_p = (gauge_p + 1) % 2

            # p round
            rho, pgp = sBs_p_evolve(rho, system_dict, gauge=gauge_p)
            # rhof = proj_list[bitstring_arr[idx_, p1[idx_r]]] * rho # measured
            rhof = rho #autonomous
            rho = qu.tensor(rho_plus, rhof.ptrace(1))
            # gauge update
            gauge_q = (gauge_q + 1) % 2

        p_arr[idx_] = np.real(rhof.tr())
        
    rhof = 0 # to avoid collapsing the memory
    return rhof, p_arr

def sBs_stabilization_pbitspath(v: float, system_dict: Dict):
    """
    Evolution of the sensor state with rounds of stabilisation (with sBs) for different q-bitstring possible. That is, the p quadrature is autonomous, and the q is measured.
    This is done to build the estimators up to 8 bits per quadrature. 
    Input:
    - v : Displacement value
    - system_dict : dictionary containing all the parameters defining the system.
    Return:
    - p_arr_r : probability list as function of rounds
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

    # Initial sensor state
    rho_init = system_dict["simulations"]["sensor"]

    # Number of rounds
    r = system_dict["simulations"]["r"]
    l = system_dict["state_params"]["l"]

    bitstring_arr = np.array(list(itertools.product([0, 1], repeat= r))) #r instead of 2r
    p_arr = np.zeros(len(bitstring_arr))
    
    beta = (system_dict['simulations']['p_0'] * l + 1j * v)/np.sqrt(2)#note here p0 stands for q0, is not changed simply to keep the previous style of the code
    displaced_rho_0 = (
            Displacement(beta, system_dict)
            * rho_init
            * Displacement(beta, system_dict).dag()
        )


    for idx_, y_ in enumerate(bitstring_arr):
        q1 = np.arange(0, r, 1)
        # p1 = np.arange(1, 2 * r + 1, 2) #no needed for autonomous
        
        rho = qu.tensor(rho_plus, displaced_rho_0)

        gauge_q = 0
        gauge_p = 0

        for idx_r, round in enumerate(np.linspace(0, r - 1, r)):
            # q round
            rho, pgq = sBs_q_evolve(rho, system_dict, gauge=gauge_q)
            # rhof = proj_list[bitstring_arr[idx_, q1[idx_r]]] * rho
            rhof = rho #autonomous
            rho = qu.tensor(rho_plus, rhof.ptrace(1))
            # gauge update
            gauge_p = (gauge_p + 1) % 2


            # p round
            rho, pgp = sBs_p_evolve(rho, system_dict, gauge=gauge_p)
            rhof = proj_list[bitstring_arr[idx_, q1[idx_r]]] * rho # measured
            # rhof = proj_list[0] * rho + proj_list[1] * rho #autonomous
            rho = qu.tensor(rho_plus, rhof.ptrace(1))

            # gauge update
            gauge_q = (gauge_q + 1) % 2

        p_arr[idx_] = np.real(rhof.tr())
    
    rhof = 0 # to avoid collapsing the memory
    return rhof, p_arr



def sBs_stabilization_qbitspath_withfinalqp(v: float, system_dict: Dict):
    """
    Evolution of the sensor state with rounds of stabilisation (with sBs) for different q-bitstring possible. That is, the p quadrature is autonomous, and the q is measured.
    This is done to build the estimators up to 8 bits per quadrature. 
    Input:
    - v : Displacement value
    - system_dict : dictionary containing all the parameters defining the system.
    Return:
    - p_arr_r : probability list as function of rounds
    - qf_arr : final q quadrature expectation value
    - pf_arr : final p quadrature expectation value
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
    a_op = system_dict["operators"]["cavity"]["a"]
    q_op = (a_op+ a_op.dag())/np.sqrt(2)
    p_op = -1j*(a_op - a_op.dag())/np.sqrt(2)

    proj_list = [g_proj, e_proj]

    # Initial sensor state
    rho_init = system_dict["simulations"]["sensor"]

    # Number of rounds
    r = system_dict["simulations"]["r"]
    l = system_dict["state_params"]["l"]

    bitstring_arr = np.array(list(itertools.product([0, 1], repeat= r))) #r instead of 2r
    p_arr = np.zeros(len(bitstring_arr))
    qf_arr, pf_arr = np.zeros(len(bitstring_arr)), np.zeros(len(bitstring_arr))

    displaced_rho_0 = (
            Displacement((v + 1.j * system_dict['simulations']['p_0'] * l) / np.sqrt(2), system_dict)
            * rho_init
            * Displacement((v + 1.j * system_dict['simulations']['p_0'] * l) / np.sqrt(2), system_dict).dag()
        )

    for idx_, y_ in enumerate(bitstring_arr):
        q1 = np.arange(0, r, 1)
        # p1 = np.arange(1, 2 * r + 1, 2) #no needed for autonomous
        rho = qu.tensor(rho_plus, displaced_rho_0)
        gauge_q = 0
        gauge_p = 0

        for idx_r, round in enumerate(np.linspace(0, r - 1, r)):
            # q round
            rho, pgq = sBs_q_evolve(rho, system_dict, gauge=gauge_q)
            rhof = proj_list[bitstring_arr[idx_, q1[idx_r]]] * rho
            rho = qu.tensor(rho_plus, rhof.ptrace(1))
            # gauge update
            gauge_p = (gauge_p + 1) % 2

            # p round
            rho, pgp = sBs_p_evolve(rho, system_dict, gauge=gauge_p)
            # rhof = proj_list[bitstring_arr[idx_, p1[idx_r]]] * rho # measured
            rhof = rho #autonomous
            rho = qu.tensor(rho_plus, rhof.ptrace(1))
            # gauge update
            gauge_q = (gauge_q + 1) % 2

        p_arr[idx_] = np.real(rhof.tr())

        rhof = rhof.ptrace(1)
        rhof = rhof/rhof.tr()
        qf_arr[idx_] = np.real(qu.expect(q_op, rhof))
        pf_arr[idx_] = np.real(qu.expect(p_op, rhof))
                
    rhof = 0 # to avoid collapsing the memory
    return rhof, p_arr, qf_arr, pf_arr


def sBs_stabilization_pbitspath_withfinalqp(v: float, system_dict: Dict):
    """
    Evolution of the sensor state with rounds of stabilisation (with sBs) for different q-bitstring possible. That is, the p quadrature is autonomous, and the q is measured.
    This is done to build the estimators up to 8 bits per quadrature. 
    Input:
    - v : Displacement value
    - system_dict : dictionary containing all the parameters defining the system.
    Return:
    - p_arr_r : probability list as function of rounds
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
    a_op = system_dict["operators"]["cavity"]["a"]
    q_op = (a_op+ a_op.dag())/np.sqrt(2)
    p_op = -1j*(a_op - a_op.dag())/np.sqrt(2)

    proj_list = [g_proj, e_proj]

    # Initial sensor state
    rho_init = system_dict["simulations"]["sensor"]

    # Number of rounds
    r = system_dict["simulations"]["r"]
    l = system_dict["state_params"]["l"]

    bitstring_arr = np.array(list(itertools.product([0, 1], repeat= r))) #r instead of 2r
    p_arr = np.zeros(len(bitstring_arr))
    qf_arr, pf_arr = np.zeros(len(bitstring_arr)), np.zeros(len(bitstring_arr))

    beta = (system_dict['simulations']['p_0'] * l + 1j * v)/np.sqrt(2)#note here p0 stands for q0, is not changed simply to keep the previous style of the code
    displaced_rho_0 = (
            Displacement(beta, system_dict)
            * rho_init
            * Displacement(beta, system_dict).dag()
        )


    for idx_, y_ in enumerate(bitstring_arr):
        q1 = np.arange(0, r, 1)
        # p1 = np.arange(1, 2 * r + 1, 2) #no needed for autonomous
        
        rho = qu.tensor(rho_plus, displaced_rho_0)

        gauge_q = 0
        gauge_p = 0

        for idx_r, round in enumerate(np.linspace(0, r - 1, r)):
            # q round
            rho, pgq = sBs_q_evolve(rho, system_dict, gauge=gauge_q)
            # rhof = proj_list[bitstring_arr[idx_, q1[idx_r]]] * rho
            rhof = rho #autonomous
            rho = qu.tensor(rho_plus, rhof.ptrace(1))
            # gauge update
            gauge_p = (gauge_p + 1) % 2


            # p round
            rho, pgp = sBs_p_evolve(rho, system_dict, gauge=gauge_p)
            rhof = proj_list[bitstring_arr[idx_, q1[idx_r]]] * rho # measured
            # rhof = proj_list[0] * rho + proj_list[1] * rho #autonomous
            rho = qu.tensor(rho_plus, rhof.ptrace(1))

            # gauge update
            gauge_q = (gauge_q + 1) % 2

        p_arr[idx_] = np.real(rhof.tr())
        
        rhof = rhof.ptrace(1)
        rhof = rhof/rhof.tr()
        qf_arr[idx_] = np.real(qu.expect(q_op, rhof))
        pf_arr[idx_] = np.real(qu.expect(p_op, rhof))

    rhof = 0 # to avoid collapsing the memory
    return rhof, p_arr, qf_arr, pf_arr



def sBs_stabilization_autonomous(v: float, system_dict: Dict):
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

    # Initial sensor state
    rho_init = system_dict["simulations"]["sensor"]
    l = system_dict["state_params"]["l"]

    # Number of rounds
    M = system_dict["simulations"]["M"] #note the addition of M in compared with previous sBs functions. 
    beta = (system_dict['simulations']['p_0'] * l + 1j * v)/np.sqrt(2)#note here p0 stands for q0, is not changed simply to keep the previous style of the code
    
    displaced_rho_0 = (
            Displacement(beta, system_dict)
            * rho_init
            * Displacement(beta, system_dict).dag()
        )
    rho = qu.tensor(rho_plus, displaced_rho_0)


    l = np.sqrt(2 * np.pi)
    Ndim = 140
    a_op = qu.destroy(Ndim)
    q_op = (a_op + a_op.dag())/np.sqrt(2)
    p_op = -1j*(a_op - a_op.dag())/np.sqrt(2)
    n_op = a_op.dag() * a_op
    Delta = 0.3
    envelope = (-Delta**2 * n_op).expm()
    T_q = (1j * l * q_op).expm()
    T_p = (1j * l * p_op).expm()
    T_qdelta = envelope * T_q * envelope.inv()
    T_pdelta = envelope * T_p * envelope.inv()
    # print('init Tq',(T_qdelta * rho_init).tr(),'init Tp',(T_pdelta * rho_init).tr())

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


def save_plot_data(
    param_filename: str,
    MSD_array: List,
    MSD_array_displacement: List,
    p_array: List,
    rho_f_array: List,
    vlist: List,
    variance_list: List,
    system_dict: Dict,
    results_options: Dict,
) -> None:
    # Print results
    if results_options["print_results"]:
        print("-----------------------------------------------")
        print("---> Simulations sBs with noise ")
        print("-----------------------------------------------")
        print("---> Input parameters")
        print(
            "Number of rounds                 : {}".format(
                system_dict["simulations"]["r"]
            )
        )
        print(
            "System sdimensions (N_c, N_q)    : ({},{})".format(
                system_dict["dimensions"]["cavity"], system_dict["dimensions"]["qubit"]
            )
        )
        print(
            "Delta                            : {:.2f}".format(
                system_dict["state_params"]["delta"]
            )
        )

        print("---> Output parameters")
        print(
            "Optimal MSD/variance             : {:.6f}".format(
                np.min(np.sum(MSD_array, axis=1) / variance_list)
            )
        )
        print(
            "Optimal variance                 : {:.6f}".format(
                variance_list[np.argmin(np.sum(MSD_array, axis=1) / variance_list)]
            )
        )

    # Save results
    if results_options["save_data"]:
        np.savez(
            results_options["save_path_name"] + param_filename[:-5],
            # results_options["save_path_name"] + str(system_dict["simulations"]["decay_ratio"]),
            MSD_array=MSD_array,
            MSD_array_displacement=MSD_array_displacement,
            variance_list=variance_list,
            p_array=p_array,
            vlist=vlist,
            rho0_c=system_dict["simulations"]["sensor"],
            system_dict=system_dict,
            rho_f_array=rho_f_array,
        )

    # Plot results
    if results_options["generate_figure"]:
        use0Fonts(6)
        fig = plt.figure(figsize=(3.5, 2.5), dpi=182)
        plt.plot(variance_list, 
            np.sum(MSD_array, axis=1) / variance_list,
            marker=".",
            color="#215257",
            ls=" ",
            markersize=3,
        )
        plt.xlabel(r"$\nu$")
        plt.ylabel(r"MSD/$\nu$")
        plt.tight_layout()
        if results_options["save_figure"]:
            plt.savefig("../Results/Data - MSD - " + param_filename[:-5] + ".png")
        if results_options["show_figure"]:
            plt.show()
        else:
            plt.close()

def save_plot_data_eachnoise(
    param_filename: str,
    MSD_array: List,
    MSD_array_displacement: List,
    p_array: List,
    rho_f_array: List,
    vlist: List,
    variance_list: List,
    system_dict: Dict,
    results_options: Dict,
    tag:str,
) -> None:
    # Print results
    if results_options["print_results"]:
        print("-----------------------------------------------")
        print("---> Simulations sBs with noise ")
        print("-----------------------------------------------")
        print("---> Input parameters")
        print(
            "Number of rounds                 : {}".format(
                system_dict["simulations"]["r"]
            )
        )
        print(
            "System sdimensions (N_c, N_q)    : ({},{})".format(
                system_dict["dimensions"]["cavity"], system_dict["dimensions"]["qubit"]
            )
        )
        print(
            "Delta                            : {:.2f}".format(
                results_options["Delta"]
            )
        )

        print("---> Output parameters")
        print(
            "Optimal MSD/variance             : {:.6f}".format(
                np.min(np.sum(MSD_array, axis=1) / variance_list)
            )
        )
        print(
            "Optimal variance                 : {:.6f}".format(
                variance_list[np.argmin(np.sum(MSD_array, axis=1) / variance_list)]
            )
        )

    # Save results
    if results_options["save_data"]:
        np.savez(
            # results_options["save_path_name"] + param_filename[:-5],
            results_options["save_path_name"] + str(system_dict["simulations"]["decay_ratio"])+tag,
            MSD_array=MSD_array,
            MSD_array_displacement=MSD_array_displacement,
            variance_list=variance_list,
            p_array=p_array,
            vlist=vlist,
            rho0_c=system_dict["simulations"]["sensor"],
            system_dict=system_dict,
            rho_f_array=rho_f_array,
        )

    # Plot results
    if results_options["generate_figure"]:
        use0Fonts(6)
        fig = plt.figure(figsize=(3.5, 2.5), dpi=182)
        plt.plot(variance_list, 
            np.sum(MSD_array, axis=1) / variance_list,
            marker=".",
            color="#215257",
            ls=" ",
            markersize=3,
        )
        plt.xlabel(r"$\nu$")
        plt.ylabel(r"MSD/$\nu$")
        plt.tight_layout()
        if results_options["save_figure"]:
            plt.savefig("../Results/Data - MSD - " + param_filename[:-5] + ".png")
        if results_options["show_figure"]:
            plt.show()
        else:
            plt.close()


def save_plot_data_withfinalqp(
    param_filename: str,
    MSD_array: List,
    MSD_array_displacement: List,
    p_array: List,
    qf_array: List,
    pf_array: List,
    rho_f_array: List,
    vlist: List,
    variance_list: List,
    system_dict: Dict,
    results_options: Dict,
) -> None:
    # Print results
    if results_options["print_results"]:
        print("-----------------------------------------------")
        print("---> Simulations sBs with noise ")
        print("-----------------------------------------------")
        print("---> Input parameters")
        print(
            "Number of rounds                 : {}".format(
                system_dict["simulations"]["r"]
            )
        )
        print(
            "System sdimensions (N_c, N_q)    : ({},{})".format(
                system_dict["dimensions"]["cavity"], system_dict["dimensions"]["qubit"]
            )
        )
        print(
            "Delta                            : {:.2f}".format(
                system_dict["state_params"]["delta"]
            )
        )

        print("---> Output parameters")
        print(
            "Optimal MSD/variance             : {:.6f}".format(
                np.min(np.sum(MSD_array, axis=1) / variance_list)
            )
        )
        print(
            "Optimal variance                 : {:.6f}".format(
                variance_list[np.argmin(np.sum(MSD_array, axis=1) / variance_list)]
            )
        )

    # Save results
    if results_options["save_data"]:
        np.savez(
            results_options["save_path_name"] + param_filename[:-5],
            # results_options["save_path_name"] + str(system_dict["simulations"]["decay_ratio"]),
            MSD_array=MSD_array,
            MSD_array_displacement=MSD_array_displacement,
            variance_list=variance_list,
            p_array=p_array,
            qf_array=qf_array,
            pf_array=pf_array,
            vlist=vlist,
            rho0_c=system_dict["simulations"]["sensor"],
            system_dict=system_dict,
            rho_f_array=rho_f_array,
        )

    # Plot results
    if results_options["generate_figure"]:
        use0Fonts(6)
        fig = plt.figure(figsize=(3.5, 2.5), dpi=182)
        plt.plot(variance_list, 
            np.sum(MSD_array, axis=1) / variance_list,
            marker=".",
            color="#215257",
            ls=" ",
            markersize=3,
        )
        plt.xlabel(r"$\nu$")
        plt.ylabel(r"MSD/$\nu$")
        plt.tight_layout()
        if results_options["save_figure"]:
            plt.savefig("../Results/Data - MSD - " + param_filename[:-5] + ".png")
        if results_options["show_figure"]:
            plt.show()
        else:
            plt.close()

def save_backaction(
    param_filename: str,
    system_dict: Dict,
    results_options: Dict,
    
) -> None:
    # Save results
    if results_options["save_data"]:
        np.savez(
            results_options["save_path_name"] + param_filename[:-5],
            # results_options["save_path_name"] + str(system_dict["simulations"]["decay_ratio"]),
            MSD_array=MSD_array,
            MSD_array_displacement=MSD_array_displacement,
            variance_list=variance_list,
            p_array=p_array,
            qf_array=qf_array,
            pf_array=pf_array,
            vlist=vlist,
            rho0_c=system_dict["simulations"]["sensor"],
            system_dict=system_dict,
            rho_f_array=rho_f_array,
        )
    