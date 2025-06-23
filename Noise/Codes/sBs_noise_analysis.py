import qutip as qu
from qutip import *
import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import *
import os
# from multiprocessing import Process
from pathlib import Path
import yaml
from tqdm import tqdm
import importlib
import general_functions
importlib.reload(general_functions)

from general_functions import (
    MSD_v,
    MSD_v_onlyqorp,
    sBs_stabilization,
    save_plot_data,
    save_plot_data_withfinalqp,
    sBs_stabilization_qbitspath,
    sBs_stabilization_pbitspath,
    sBs_stabilization_qbitspath_withfinalqp,
    sBs_stabilization_pbitspath_withfinalqp,
    sBs_stabilization_autonomous,
)

import backaction_noise
importlib.reload(backaction_noise)
from backaction_noise import *

script_directory = Path(os.path.dirname(os.path.abspath(__file__)))


def run_simulations_q(param_filename, param_path, results_options):
    params = yaml.safe_load(Path(param_path + param_filename).read_text())
    r = params["simulations"]["r"]
    delta = params["state_params"]["delta"]
    l = eval(params["state_params"]["l"])
    t_cd_B = params["simulations"]["t_CD_B"]

    N_c = params["dimensions"]["cavity"]
    N_q = params["dimensions"]["qubit"]
    qubit_T1 = params["decays"]["qubit_T1"]
    qubit_T2 = params["decays"]["qubit_T2"]
    cavity_T1 = params["decays"]["cavity_T1"]
    cavity_T2 = params["decays"]["cavity_T2"]
    
    T_1q = round(qubit_T1/ t_cd_B, 5)
    T_2q = round(qubit_T2/ t_cd_B, 5)
    T_1c = round(cavity_T1/ t_cd_B, 5)
    T_2c = round(cavity_T2/ t_cd_B, 5)

    T_qphi = 1 / (1 / T_2q - 1 / (2 * T_1q))
    T_cphi = 1 / (1 / T_2c - 1 / (2 * T_1c))

    init_state_dict = np.load(
                "../Results/Input_states/coherent_sensor_state_delta_{}-Nc{}-{}.npz".format(int(100 * delta), N_c, params["decays"]["ratio"])
            )
    init_state = init_state_dict['rho_f_array'][0]

    system_dict = {
        "dimensions": {"qubit": N_q, "cavity": N_c,},
        "operators": {
            "qubit": {
                "sx": qu.sigmax(),
                "sy": qu.sigmay(),
                "sz": qu.sigmaz(),
                "sm": qu.sigmam(),
            },
            "cavity": {"a": qu.destroy(N_c)},
        },
        "state_params": {"l": l, "delta": delta},
        "timings": {"t_B": np.linspace(0, 1.0, 51),},
        "c_ops": [
            qu.tensor(qu.sigmam(), qu.qeye(N_c)) / np.sqrt(T_1q),
            qu.tensor(qu.qeye(N_q), qu.destroy(N_c)) / np.sqrt(T_1c),
            # qu.tensor(qu.qeye(N_q) - qu.sigmaz(), qu.qeye(N_c)) / np.sqrt(2 * T_qphi),
            qu.tensor(qu.sigmaz(), qu.qeye(N_c)) / np.sqrt(2 * T_qphi),
            qu.tensor(qu.qeye(N_q), np.sqrt(2) * qu.destroy(N_c).dag() * qu.destroy(N_c)) / np.sqrt(T_cphi),
        ],
        "simulations": {
            "r": r,
            # "sensor": qu.qload(
            #     "../Input_states/sensor_state_delta-{}-Nc{}".format(int(100 * delta), N_c)
            # ),
            "sensor": qu.Qobj(init_state),
            'p_0': l * params['simulations']['p_0'],
        },
    }

    # defining the displacement list
    vlist = np.linspace(
        -params["simulations"]["v_lim"] * l,
        params["simulations"]["v_lim"] * l,
        params["simulations"]["n_v"],
    )

    p_array = np.empty((len(vlist), 2 ** (2 * r)))
    rho_f_array = np.empty((len(vlist), N_c, N_c), dtype=complex)

    # runing simulations
    p_array = np.array(p_array)
    parallel = Parallel(n_jobs=params["simulations"]["n_jobs"])
    output_generator = parallel(delayed(sBs_stabilization_q)(v, system_dict) for v in tqdm(vlist))

    for idx, v in enumerate(vlist):
        p_array[idx, :] = output_generator[idx][1]
        # rho_f_array[idx, :, :] = (output_generator[idx][0]).ptrace(1)


    p_array = np.array(p_array)
    variance_list = np.linspace(0.01, l / 4, 101)
    MSD_array = np.empty((len(variance_list), len(vlist)))
    MSD_array_displacement = np.empty((len(variance_list), len(vlist)))
    posterior_array = np.empty((len(variance_list), 2 ** (2 * r), len(vlist)))
    for idx, var in enumerate(variance_list):
        MSD, MSD_displacement, posterior = MSD_v(p_array, vlist, var, system_dict)
        MSD_array[idx] = MSD
        MSD_array_displacement[idx] = MSD_displacement
        posterior_array[idx] = posterior

    save_plot_data(
        param_filename,
        MSD_array,
        MSD_array_displacement,
        p_array,
        rho_f_array,
        vlist,
        variance_list,
        system_dict,
        results_options,
    )
    return None

def run_simulations_p(param_filename, param_path, results_options):
    params = yaml.safe_load(Path(param_path + param_filename).read_text())
    r = params["simulations"]["r"]
    delta = params["state_params"]["delta"]
    l = eval(params["state_params"]["l"])
    t_cd_B = params["simulations"]["t_CD_B"]

    N_c = params["dimensions"]["cavity"]
    N_q = params["dimensions"]["qubit"]
    qubit_T1 = params["decays"]["qubit_T1"]
    qubit_T2 = params["decays"]["qubit_T2"]
    cavity_T1 = params["decays"]["cavity_T1"]
    cavity_T2 = params["decays"]["cavity_T2"]
    
    T_1q = round(qubit_T1/ t_cd_B, 5)
    T_2q = round(qubit_T2/ t_cd_B, 5)
    T_1c = round(cavity_T1/ t_cd_B, 5)
    T_2c = round(cavity_T2/ t_cd_B, 5)

    T_qphi = 1 / (1 / T_2q - 1 / (2 * T_1q))
    T_cphi = 1 / (1 / T_2c - 1 / (2 * T_1c))

    init_state_dict = np.load(
                "../Results/Input_states/coherent_sensor_state_delta_{}-Nc{}-{}.npz".format(int(100 * delta), N_c, params["decays"]["ratio"])
            )
    init_state = init_state_dict['rho_f_array'][0]

    system_dict = {
        "dimensions": {"qubit": N_q, "cavity": N_c,},
        "operators": {
            "qubit": {
                "sx": qu.sigmax(),
                "sy": qu.sigmay(),
                "sz": qu.sigmaz(),
                "sm": qu.sigmam(),
            },
            "cavity": {"a": qu.destroy(N_c)},
        },
        "state_params": {"l": l, "delta": delta},
        "timings": {"t_B": np.linspace(0, 1.0, 51),},
        "c_ops": [
            qu.tensor(qu.sigmam(), qu.qeye(N_c)) / np.sqrt(T_1q),
            qu.tensor(qu.qeye(N_q), qu.destroy(N_c)) / np.sqrt(T_1c),
            # qu.tensor(qu.qeye(N_q) - qu.sigmaz(), qu.qeye(N_c)) / np.sqrt(2 * T_qphi),
            qu.tensor(qu.sigmaz(), qu.qeye(N_c)) / np.sqrt(2 * T_qphi),
            qu.tensor(qu.qeye(N_q), np.sqrt(2) * qu.destroy(N_c).dag() * qu.destroy(N_c)) / np.sqrt(T_cphi),
        ],
        "simulations": {
            "r": r,
            # "sensor": qu.qload(
            #     "../Input_states/sensor_state_delta-{}-Nc{}".format(int(100 * delta), N_c)
            # ),
            "sensor": qu.Qobj(init_state),
            'p_0': l * params['simulations']['p_0'],
        },
    }

    # defining the displacement list
    vlist = np.linspace(
        -params["simulations"]["v_lim"] * l,
        params["simulations"]["v_lim"] * l,
        params["simulations"]["n_v"],
    )

    p_array = np.empty((len(vlist), 2 ** (2 * r)))
    rho_f_array = np.empty((len(vlist), N_c, N_c), dtype=complex)

    # runing simulations
    p_array = np.array(p_array)
    parallel = Parallel(n_jobs=params["simulations"]["n_jobs"])
    output_generator = parallel(delayed(sBs_stabilization_p)(v, system_dict) for v in tqdm(vlist))

    for idx, v in enumerate(vlist):
        p_array[idx, :] = output_generator[idx][1]
        # rho_f_array[idx, :, :] = (output_generator[idx][0]).ptrace(1)


    p_array = np.array(p_array)
    variance_list = np.linspace(0.01, l / 4, 101)
    MSD_array = np.empty((len(variance_list), len(vlist)))
    MSD_array_displacement = np.empty((len(variance_list), len(vlist)))
    posterior_array = np.empty((len(variance_list), 2 ** (2 * r), len(vlist)))
    for idx, var in enumerate(variance_list):
        MSD, MSD_displacement, posterior = MSD_v(p_array, vlist, var, system_dict)
        MSD_array[idx] = MSD
        MSD_array_displacement[idx] = MSD_displacement
        posterior_array[idx] = posterior

    save_plot_data(
        param_filename,
        MSD_array,
        MSD_array_displacement,
        p_array,
        rho_f_array,
        vlist,
        variance_list,
        system_dict,
        results_options,
    )
    return None



def run_simulations_qpath(param_filename, param_path, results_options):
    params = yaml.safe_load(Path(param_path + param_filename).read_text())
    r = params["simulations"]["r"]
    delta = params["state_params"]["delta"]
    l = eval(params["state_params"]["l"])
    t_cd_B = params["simulations"]["t_CD_B"]

    N_c = params["dimensions"]["cavity"]
    N_q = params["dimensions"]["qubit"]
    qubit_T1 = params["decays"]["qubit_T1"]
    qubit_T2 = params["decays"]["qubit_T2"]
    cavity_T1 = params["decays"]["cavity_T1"]
    cavity_T2 = params["decays"]["cavity_T2"]
    
    T_1q = round(qubit_T1/ t_cd_B, 5)
    T_2q = round(qubit_T2/ t_cd_B, 5)
    T_1c = round(cavity_T1/ t_cd_B, 5)
    T_2c = round(cavity_T2/ t_cd_B, 5)

    T_qphi = 1 / (1 / T_2q - 1 / (2 * T_1q))
    T_cphi = 1 / (1 / T_2c - 1 / (2 * T_1c))

    init_state_dict = np.load(
                "../Results/Input_states/coherent_sensor_state_delta_{}-Nc{}-{}.npz".format(int(100 * delta), N_c, params["decays"]["ratio"])
            )
    init_state = init_state_dict['rho_f_array'][0]

    system_dict = {
        "dimensions": {"qubit": N_q, "cavity": N_c,},
        "operators": {
            "qubit": {
                "sx": qu.sigmax(),
                "sy": qu.sigmay(),
                "sz": qu.sigmaz(),
                "sm": qu.sigmam(),
            },
            "cavity": {"a": qu.destroy(N_c)},
        },
        "state_params": {"l": l, "delta": delta},
        "timings": {"t_B": np.linspace(0, 1.0, 51),},
        "c_ops": [
            qu.tensor(qu.sigmam(), qu.qeye(N_c)) / np.sqrt(T_1q),
            qu.tensor(qu.qeye(N_q), qu.destroy(N_c)) / np.sqrt(T_1c),
            # qu.tensor(qu.qeye(N_q) - qu.sigmaz(), qu.qeye(N_c)) / np.sqrt(2 * T_qphi),
            qu.tensor(qu.sigmaz(), qu.qeye(N_c)) / np.sqrt(2 * T_qphi),
            qu.tensor(qu.qeye(N_q), np.sqrt(2) * qu.destroy(N_c).dag() * qu.destroy(N_c)) / np.sqrt(T_cphi),
        ],
        "simulations": {
            "r": r,
            "sensor": qu.Qobj(init_state),
            'p_0': l * params['simulations']['p_0'],
        },
    }

    # defining the displacement list
    vlist = np.linspace(
        -params["simulations"]["v_lim"] * l,
        params["simulations"]["v_lim"] * l,
        params["simulations"]["n_v"],
    )

    # print('tB',1.0)

    p_array = np.empty((len(vlist), 2 ** (r)))
    rho_f_array = np.empty((len(vlist), N_c, N_c), dtype=complex)

    # runing simulations
    p_array = np.array(p_array)
    parallel = Parallel(n_jobs=params["simulations"]["n_jobs"])
    output_generator = parallel(delayed(sBs_stabilization_qbitspath)(v, system_dict) for v in tqdm(vlist))

    for idx, v in enumerate(vlist):
        p_array[idx, :] = output_generator[idx][1] #p(b|q0), with shape [q0][bitstring]
        # rho_f_array[idx, :, :] = (output_generator[idx][0]).ptrace(1)

    # print('p_array', p_array.shape)
    p_array = np.array(p_array)
    variance_list = np.linspace(0.01, l / 4, 101)
    MSD_array = np.empty((len(variance_list), len(vlist)))
    MSD_array_displacement = np.empty((len(variance_list), len(vlist)))
    posterior_array = np.empty((len(variance_list), 2 ** (r), len(vlist)))
    for idx, var in enumerate(variance_list):
        MSD, MSD_displacement, posterior = MSD_v_onlyqorp(p_array, vlist, var, system_dict)
        MSD_array[idx] = MSD
        MSD_array_displacement[idx] = MSD_displacement
        posterior_array[idx] = posterior

    save_plot_data(
        param_filename,
        MSD_array,
        MSD_array_displacement,
        p_array,
        rho_f_array,
        vlist,
        variance_list,
        system_dict,
        results_options,
    )
    return None

def run_simulations_ppath(param_filename, param_path, results_options):
    params = yaml.safe_load(Path(param_path + param_filename).read_text())
    r = params["simulations"]["r"]
    delta = params["state_params"]["delta"]
    l = eval(params["state_params"]["l"])
    t_cd_B = params["simulations"]["t_CD_B"]

    N_c = params["dimensions"]["cavity"]
    N_q = params["dimensions"]["qubit"]
    qubit_T1 = params["decays"]["qubit_T1"]
    qubit_T2 = params["decays"]["qubit_T2"]
    cavity_T1 = params["decays"]["cavity_T1"]
    cavity_T2 = params["decays"]["cavity_T2"]
    
    T_1q = round(qubit_T1/ t_cd_B, 5)
    T_2q = round(qubit_T2/ t_cd_B, 5)
    T_1c = round(cavity_T1/ t_cd_B, 5)
    T_2c = round(cavity_T2/ t_cd_B, 5)

    T_qphi = 1 / (1 / T_2q - 1 / (2 * T_1q))
    T_cphi = 1 / (1 / T_2c - 1 / (2 * T_1c))

    file_number = result_options["file_number"]

    if file_number in results_options["to_run_dict"]["ideal_state_noisy_metrology"]:
        init_state_path =  "../Input_states/sensor_state_delta-{0:.0f}-Nc140".format(delta*100)
        init_state = qu.qload(init_state_path)
        c_ops = [
            qu.tensor(qu.sigmam(), qu.qeye(N_c)) / np.sqrt(T_1q),
            qu.tensor(qu.qeye(N_q), qu.destroy(N_c)) / np.sqrt(T_1c),
            # qu.tensor(qu.qeye(N_q) - qu.sigmaz(), qu.qeye(N_c)) / np.sqrt(2 * T_qphi),
            qu.tensor(qu.sigmaz(), qu.qeye(N_c)) / np.sqrt(2 * T_qphi),
            qu.tensor(qu.qeye(N_q), np.sqrt(2) * qu.destroy(N_c).dag() * qu.destroy(N_c)) / np.sqrt(T_cphi),
        ]
        print('Using ideal state with noisy metrology ', file_number)
    elif file_number in results_options["to_run_dict"]["noisy_state_ideal_metrology"]:
        init_state_dict = np.load(
                    "../Results/Input_states/coherent_sensor_state_delta_{}-Nc{}-{}.npz".format(int(100 * delta), N_c, params["decays"]["ratio"])
                )
        init_state = init_state_dict['rho_f_array'][0]
        c_ops = []
        print('Using noisy state with ideal metrology ', file_number)
    else:
        init_state_dict = np.load(
                    "../Results/Input_states/coherent_sensor_state_delta_{}-Nc{}-{}.npz".format(int(100 * delta), N_c, params["decays"]["ratio"])
                )
        c_ops = [
            qu.tensor(qu.sigmam(), qu.qeye(N_c)) / np.sqrt(T_1q),
            qu.tensor(qu.qeye(N_q), qu.destroy(N_c)) / np.sqrt(T_1c),
            # qu.tensor(qu.qeye(N_q) - qu.sigmaz(), qu.qeye(N_c)) / np.sqrt(2 * T_qphi),
            qu.tensor(qu.sigmaz(), qu.qeye(N_c)) / np.sqrt(2 * T_qphi),
            qu.tensor(qu.qeye(N_q), np.sqrt(2) * qu.destroy(N_c).dag() * qu.destroy(N_c)) / np.sqrt(T_cphi),
        ]

    system_dict = {
        "dimensions": {"qubit": N_q, "cavity": N_c,},
        "operators": {
            "qubit": {
                "sx": qu.sigmax(),
                "sy": qu.sigmay(),
                "sz": qu.sigmaz(),
                "sm": qu.sigmam(),
            },
            "cavity": {"a": qu.destroy(N_c)},
        },
        "state_params": {"l": l, "delta": delta},
        "timings": {"t_B": np.linspace(0, 1.0, 51),},
        "c_ops": c_ops,
        "simulations": {
            "r": r,
            "sensor": qu.Qobj(init_state),
            'p_0': l * params['simulations']['p_0'],
        },
    }

    # defining the displacement list
    vlist = np.linspace(
        -params["simulations"]["v_lim"] * l,
        params["simulations"]["v_lim"] * l,
        params["simulations"]["n_v"],
    )

    p_array = np.empty((len(vlist), 2 ** (r)))
    rho_f_array = np.empty((len(vlist), N_c, N_c), dtype=complex)

    # runing simulations
    p_array = np.array(p_array)
    parallel = Parallel(n_jobs=params["simulations"]["n_jobs"])
    output_generator = parallel(delayed(sBs_stabilization_pbitspath)(v, system_dict) for v in tqdm(vlist))

    for idx, v in enumerate(vlist):
        p_array[idx, :] = output_generator[idx][1]
        # rho_f_array[idx, :, :] = (output_generator[idx][0]).ptrace(1)

    # print('p_array', p_array.shape)
    p_array = np.array(p_array)
    variance_list = np.linspace(0.01, l / 4, 101)
    MSD_array = np.empty((len(variance_list), len(vlist)))
    MSD_array_displacement = np.empty((len(variance_list), len(vlist)))
    posterior_array = np.empty((len(variance_list), 2 ** (r), len(vlist)))
    for idx, var in enumerate(variance_list):
        MSD, MSD_displacement, posterior = MSD_v_onlyqorp(p_array, vlist, var, system_dict)
        MSD_array[idx] = MSD
        MSD_array_displacement[idx] = MSD_displacement
        posterior_array[idx] = posterior

    save_plot_data(
        param_filename,
        MSD_array,
        MSD_array_displacement,
        p_array,
        rho_f_array,
        vlist,
        variance_list,
        system_dict,
        results_options,
    )
    return None


def run_simulations_qpath_withfinalqp(param_filename, param_path, results_options):
    params = yaml.safe_load(Path(param_path + param_filename).read_text())
    r = params["simulations"]["r"]
    delta = params["state_params"]["delta"]
    l = eval(params["state_params"]["l"])
    t_cd_B = params["simulations"]["t_CD_B"]

    N_c = params["dimensions"]["cavity"]
    N_q = params["dimensions"]["qubit"]
    qubit_T1 = params["decays"]["qubit_T1"]
    qubit_T2 = params["decays"]["qubit_T2"]
    cavity_T1 = params["decays"]["cavity_T1"]
    cavity_T2 = params["decays"]["cavity_T2"]
    
    T_1q = round(qubit_T1/ t_cd_B, 5)
    T_2q = round(qubit_T2/ t_cd_B, 5)
    T_1c = round(cavity_T1/ t_cd_B, 5)
    T_2c = round(cavity_T2/ t_cd_B, 5)

    T_qphi = 1 / (1 / T_2q - 1 / (2 * T_1q))
    T_cphi = 1 / (1 / T_2c - 1 / (2 * T_1c))

    file_number = results_options["file_number"]

    if file_number in results_options["to_run_dict"]["ideal_state_noisy_metrology"]:
        init_state_path =  "../Input_states/sensor_state_delta-{0:.0f}-Nc140".format(delta*100)
        init_state = qu.qload(init_state_path)
        c_ops = [
            qu.tensor(qu.sigmam(), qu.qeye(N_c)) / np.sqrt(T_1q),
            qu.tensor(qu.qeye(N_q), qu.destroy(N_c)) / np.sqrt(T_1c),
            # qu.tensor(qu.qeye(N_q) - qu.sigmaz(), qu.qeye(N_c)) / np.sqrt(2 * T_qphi),
            qu.tensor(qu.sigmaz(), qu.qeye(N_c)) / np.sqrt(2 * T_qphi),
            qu.tensor(qu.qeye(N_q), np.sqrt(2) * qu.destroy(N_c).dag() * qu.destroy(N_c)) / np.sqrt(T_cphi),
        ]
        print('Using ideal state with noisy metrology ', file_number)
    elif file_number in results_options["to_run_dict"]["noisy_state_ideal_metrology"]:
        init_state_dict = np.load(
                    "../Results/Input_states/coherent_sensor_state_delta_{}-Nc{}-{}.npz".format(int(100 * delta), N_c, params["decays"]["ratio"])
                )
        init_state = init_state_dict['rho_f_array'][0]
        c_ops = []
        print('Using noisy state with ideal metrology ', file_number)
    else:
        init_state_dict = np.load(
                    "../Results/Input_states/coherent_sensor_state_delta_{}-Nc{}-{}.npz".format(int(100 * delta), N_c, params["decays"]["ratio"])
                )
        init_state = init_state_dict['rho_f_array'][0]
        c_ops = [
            qu.tensor(qu.sigmam(), qu.qeye(N_c)) / np.sqrt(T_1q),
            qu.tensor(qu.qeye(N_q), qu.destroy(N_c)) / np.sqrt(T_1c),
            # qu.tensor(qu.qeye(N_q) - qu.sigmaz(), qu.qeye(N_c)) / np.sqrt(2 * T_qphi),
            qu.tensor(qu.sigmaz(), qu.qeye(N_c)) / np.sqrt(2 * T_qphi),
            qu.tensor(qu.qeye(N_q), np.sqrt(2) * qu.destroy(N_c).dag() * qu.destroy(N_c)) / np.sqrt(T_cphi),
        ]
        print('Using noisy state with noisy metrology ', file_number)

    system_dict = {
        "dimensions": {"qubit": N_q, "cavity": N_c,},
        "operators": {
            "qubit": {
                "sx": qu.sigmax(),
                "sy": qu.sigmay(),
                "sz": qu.sigmaz(),
                "sm": qu.sigmam(),
            },
            "cavity": {"a": qu.destroy(N_c)},
        },
        "state_params": {"l": l, "delta": delta},
        "timings": {"t_B": np.linspace(0, 1.0, 51),},
        "c_ops": c_ops,
        "simulations": {
            "r": r,
            "sensor": qu.Qobj(init_state),
            'p_0': l * params['simulations']['p_0'],
        },
    }

    # defining the displacement list
    vlist = np.linspace(
        -params["simulations"]["v_lim"] * l,
        params["simulations"]["v_lim"] * l,
        params["simulations"]["n_v"],
    )

    # print('tB',1.0)

    p_array = np.empty((len(vlist), 2 ** (r)))
    qf_array = np.empty((len(vlist), 2 ** (r)))
    pf_array = np.empty((len(vlist), 2 ** (r)))
    rho_f_array = np.empty((len(vlist), N_c, N_c), dtype=complex)

    # runing simulations
    p_array = np.array(p_array)
    parallel = Parallel(n_jobs=params["simulations"]["n_jobs"])
    output_generator = parallel(delayed(sBs_stabilization_qbitspath_withfinalqp)(v, system_dict) for v in tqdm(vlist))

    for idx, v in enumerate(vlist):
        p_array[idx, :] = output_generator[idx][1] #p(b|q0), with shape [q0][bitstring]
        qf_array[idx, :] = output_generator[idx][2] #p(b|p0), with shape [q0][bitstring]
        pf_array[idx, :] = output_generator[idx][3] #p(b|p0), with shape [p0][bitstring]
        # rho_f_array[idx, :, :] = (output_generator[idx][0]).ptrace(1)

    # print('p_array', p_array.shape)
    p_array = np.array(p_array)
    variance_list = np.linspace(0.01, l / 4, 101)
    MSD_array = np.empty((len(variance_list), len(vlist)))
    MSD_array_displacement = np.empty((len(variance_list), len(vlist)))
    posterior_array = np.empty((len(variance_list), 2 ** (r), len(vlist)))
    for idx, var in enumerate(variance_list):
        MSD, MSD_displacement, posterior = MSD_v_onlyqorp(p_array, vlist, var, system_dict)
        MSD_array[idx] = MSD
        MSD_array_displacement[idx] = MSD_displacement
        posterior_array[idx] = posterior

    save_plot_data_withfinalqp(
        param_filename,
        MSD_array,
        MSD_array_displacement,
        p_array,
        qf_array,
        pf_array,
        rho_f_array,
        vlist,
        variance_list,
        system_dict,
        results_options,
    )
    return None



def run_simulations_ppath_withfinalqp(param_filename, param_path, results_options):
    params = yaml.safe_load(Path(param_path + param_filename).read_text())
    r = params["simulations"]["r"]
    delta = params["state_params"]["delta"]
    l = eval(params["state_params"]["l"])
    t_cd_B = params["simulations"]["t_CD_B"]

    N_c = params["dimensions"]["cavity"]
    N_q = params["dimensions"]["qubit"]
    qubit_T1 = params["decays"]["qubit_T1"]
    qubit_T2 = params["decays"]["qubit_T2"]
    cavity_T1 = params["decays"]["cavity_T1"]
    cavity_T2 = params["decays"]["cavity_T2"]
    
    T_1q = round(qubit_T1/ t_cd_B, 5)
    T_2q = round(qubit_T2/ t_cd_B, 5)
    T_1c = round(cavity_T1/ t_cd_B, 5)
    T_2c = round(cavity_T2/ t_cd_B, 5)

    T_qphi = 1 / (1 / T_2q - 1 / (2 * T_1q))
    T_cphi = 1 / (1 / T_2c - 1 / (2 * T_1c))

    init_state_dict = np.load(
                    "../Results/Input_states/coherent_sensor_state_delta_{}-Nc{}-{}.npz".format(int(100 * delta), N_c, params["decays"]["ratio"])
                )
    init_state = init_state_dict['rho_f_array'][0]

    c_ops = [
            qu.tensor(qu.sigmam(), qu.qeye(N_c)) / np.sqrt(T_1q),
            qu.tensor(qu.qeye(N_q), qu.destroy(N_c)) / np.sqrt(T_1c),
            # qu.tensor(qu.qeye(N_q) - qu.sigmaz(), qu.qeye(N_c)) / np.sqrt(2 * T_qphi),
            qu.tensor(qu.sigmaz(), qu.qeye(N_c)) / np.sqrt(2 * T_qphi),
            qu.tensor(qu.qeye(N_q), np.sqrt(2) * qu.destroy(N_c).dag() * qu.destroy(N_c)) / np.sqrt(T_cphi),
        ]

    # name2 = 'sensor_state_delta-{0:.0f}-Nc140'.format(delta*100)
    # ideal_state = qu.qload(path_sbs_noise+'/Input_states/'+name2)
    # init_state = ideal_state


    system_dict = {
        "dimensions": {"qubit": N_q, "cavity": N_c,},
        "operators": {
            "qubit": {
                "sx": qu.sigmax(),
                "sy": qu.sigmay(),
                "sz": qu.sigmaz(),
                "sm": qu.sigmam(),
            },
            "cavity": {"a": qu.destroy(N_c)},
        },
        "state_params": {"l": l, "delta": delta},
        "timings": {"t_B": np.linspace(0, 1.0, 51),},
        "c_ops": c_ops,
        "simulations": {
            "r": r,
            "sensor": qu.Qobj(init_state),
            'p_0': l * params['simulations']['p_0'],
        },
    }

    # defining the displacement list
    vlist = np.linspace(
        -params["simulations"]["v_lim"] * l,
        params["simulations"]["v_lim"] * l,
        params["simulations"]["n_v"],
    )

    # print('tB',1.0)

    p_array = np.empty((len(vlist), 2 ** (r)))
    qf_array = np.empty((len(vlist), 2 ** (r)))
    pf_array = np.empty((len(vlist), 2 ** (r)))
    rho_f_array = np.empty((len(vlist), N_c, N_c), dtype=complex)

    # runing simulations
    p_array = np.array(p_array)
    parallel = Parallel(n_jobs=params["simulations"]["n_jobs"])
    output_generator = parallel(delayed(sBs_stabilization_pbitspath_withfinalqp)(v, system_dict) for v in tqdm(vlist))

    for idx, v in enumerate(vlist):
        p_array[idx, :] = output_generator[idx][1] #p(b|q0), with shape [q0][bitstring]
        qf_array[idx, :] = output_generator[idx][2] #p(b|p0), with shape [q0][bitstring]
        pf_array[idx, :] = output_generator[idx][3] #p(b|p0), with shape [p0][bitstring]
        # rho_f_array[idx, :, :] = (output_generator[idx][0]).ptrace(1)

    # print('p_array', p_array.shape)
    p_array = np.array(p_array)
    variance_list = np.linspace(0.01, l / 4, 101)
    MSD_array = np.empty((len(variance_list), len(vlist)))
    MSD_array_displacement = np.empty((len(variance_list), len(vlist)))
    posterior_array = np.empty((len(variance_list), 2 ** (r), len(vlist)))
    for idx, var in enumerate(variance_list):
        MSD, MSD_displacement, posterior = MSD_v_onlyqorp(p_array, vlist, var, system_dict)
        MSD_array[idx] = MSD
        MSD_array_displacement[idx] = MSD_displacement
        posterior_array[idx] = posterior

    save_plot_data_withfinalqp(
        param_filename,
        MSD_array,
        MSD_array_displacement,
        p_array,
        qf_array,
        pf_array,
        rho_f_array,
        vlist,
        variance_list,
        system_dict,
        results_options,
    )
    return None



def run_simulations_qpath_withfinalqp_eachnoise(param_filename, param_path, results_options):
    params = yaml.safe_load(Path(param_path + param_filename).read_text())
    r = params["simulations"]["r"]
    delta = params["state_params"]["delta"]
    l = eval(params["state_params"]["l"])
    t_cd_B = params["simulations"]["t_CD_B"]

    N_c = params["dimensions"]["cavity"]
    N_q = params["dimensions"]["qubit"]
    qubit_T1 = params["decays"]["qubit_T1"]
    qubit_T2 = params["decays"]["qubit_T2"]
    cavity_T1 = params["decays"]["cavity_T1"]
    cavity_T2 = params["decays"]["cavity_T2"]
    
    T_1q = round(qubit_T1/ t_cd_B, 5)
    T_2q = round(qubit_T2/ t_cd_B, 5)
    T_1c = round(cavity_T1/ t_cd_B, 5)
    T_2c = round(cavity_T2/ t_cd_B, 5)

    T_qphi = 1 / (1 / T_2q - 1 / (2 * T_1q))
    T_cphi = 1 / (1 / T_2c - 1 / (2 * T_1c))

    # init_state_dict = np.load(
    #             "../Results/Input_states/coherent_sensor_state_delta_{}-Nc{}-{}.npz".format(int(100 * delta), N_c, params["decays"]["ratio"])
    #         )
    # init_state = init_state_dict['rho_f_array'][0]

    init_state = qu.qload(results_options["init_state_path"])


    #choosing collapse operators, given by file number
    file_number = results_options["file_number"]
    if file_number in results_options["collapse_dict"]["T1c"]:
        c_ops = [qu.tensor(qu.qeye(N_q), qu.destroy(N_c)) / np.sqrt(T_1c)]
        tag = "T1c"
    elif file_number in results_options["collapse_dict"]["T2c"]:
        c_ops = [qu.tensor(qu.qeye(N_q), np.sqrt(2) * qu.destroy(N_c).dag() * qu.destroy(N_c)) / np.sqrt(T_cphi)]
        tag = "T2c"
    elif file_number in results_options["collapse_dict"]["T1q"]:
        c_ops = [qu.tensor(qu.sigmam(), qu.qeye(N_c)) / np.sqrt(T_1q)]
        tag = "T1q"
    elif file_number in results_options["collapse_dict"]["T2q"]:
        c_ops = [qu.tensor(qu.sigmaz(), qu.qeye(N_c)) / np.sqrt(2 * T_qphi)]
        tag = "T2q"
    else:
        raise ValueError("File number does not match any collapse operator category.", file_number)

    print(tag,file_number, T_1c, T_cphi, T_1q, T_qphi)

    system_dict = {
        "dimensions": {"qubit": N_q, "cavity": N_c,},
        "operators": {
            "qubit": {
                "sx": qu.sigmax(),
                "sy": qu.sigmay(),
                "sz": qu.sigmaz(),
                "sm": qu.sigmam(),
            },
            "cavity": {"a": qu.destroy(N_c)},
        },
        "state_params": {"l": l, "delta": delta},
        "timings": {"t_B": np.linspace(0, 1.0, 51),},
        "c_ops": c_ops,
        "simulations": {
            "r": r,
            "sensor": qu.Qobj(init_state),
            'p_0': l * params['simulations']['p_0'],
        },
    }

    # defining the displacement list
    vlist = np.linspace(
        -params["simulations"]["v_lim"] * l,
        params["simulations"]["v_lim"] * l,
        params["simulations"]["n_v"],
    )

    # print('tB',1.0)

    p_array = np.empty((len(vlist), 2 ** (r)))
    qf_array = np.empty((len(vlist), 2 ** (r)))
    pf_array = np.empty((len(vlist), 2 ** (r)))
    rho_f_array = np.empty((len(vlist), N_c, N_c), dtype=complex)

    # runing simulations
    p_array = np.array(p_array)
    parallel = Parallel(n_jobs=params["simulations"]["n_jobs"])
    output_generator = parallel(delayed(sBs_stabilization_qbitspath_withfinalqp)(v, system_dict) for v in tqdm(vlist))

    for idx, v in enumerate(vlist):
        p_array[idx, :] = output_generator[idx][1] #p(b|q0), with shape [q0][bitstring]
        qf_array[idx, :] = output_generator[idx][2] #p(b|p0), with shape [q0][bitstring]
        pf_array[idx, :] = output_generator[idx][3] #p(b|p0), with shape [p0][bitstring]
        # rho_f_array[idx, :, :] = (output_generator[idx][0]).ptrace(1)

    # print('p_array', p_array.shape)
    p_array = np.array(p_array)
    variance_list = np.linspace(0.01, l / 4, 101)
    MSD_array = np.empty((len(variance_list), len(vlist)))
    MSD_array_displacement = np.empty((len(variance_list), len(vlist)))
    posterior_array = np.empty((len(variance_list), 2 ** (r), len(vlist)))
    for idx, var in enumerate(variance_list):
        MSD, MSD_displacement, posterior = MSD_v_onlyqorp(p_array, vlist, var, system_dict)
        MSD_array[idx] = MSD
        MSD_array_displacement[idx] = MSD_displacement
        posterior_array[idx] = posterior

    save_plot_data_withfinalqp(
        param_filename,
        MSD_array,
        MSD_array_displacement,
        p_array,
        qf_array,
        pf_array,
        rho_f_array,
        vlist,
        variance_list,
        system_dict,
        results_options,
    )
    return None


def run_simulations_autonomous(param_filename, param_path, results_options):
    params = yaml.safe_load(Path(param_path + param_filename).read_text())
    r = params["simulations"]["r"]
    M = params["simulations"]["M"]
    delta = params["state_params"]["delta"]
    l = eval(params["state_params"]["l"])
    t_cd_B = params["simulations"]["t_CD_B"]

    N_c = params["dimensions"]["cavity"]
    N_q = params["dimensions"]["qubit"]
    qubit_T1 = params["decays"]["qubit_T1"]
    qubit_T2 = params["decays"]["qubit_T2"]
    cavity_T1 = params["decays"]["cavity_T1"]
    cavity_T2 = params["decays"]["cavity_T2"]
    
    T_1q = round(qubit_T1/ t_cd_B, 5)
    T_2q = round(qubit_T2/ t_cd_B, 5)
    T_1c = round(cavity_T1/ t_cd_B, 5)
    T_2c = round(cavity_T2/ t_cd_B, 5)

    T_qphi = 1 / (1 / T_2q - 1 / (2 * T_1q))
    T_cphi = 1 / (1 / T_2c - 1 / (2 * T_1c))

    if results_options["coherent"]:
        init_state = qu.fock_dm(N_c, 0)
        print('initial state coherent')
    else:
        print('initial state stabilized sensor')
        init_state_dict = np.load(
                "../Results/Input_states/coherent_sensor_state_delta_{}-Nc{}-{}.npz".format(int(100 * delta), N_c, params["decays"]["ratio"])
            )
        init_state = qu.Qobj(init_state_dict['rho_f_array'][0])
        

    system_dict = {
        "dimensions": {"qubit": N_q, "cavity": N_c,},
        "operators": {
            "qubit": {
                "sx": qu.sigmax(),
                "sy": qu.sigmay(),
                "sz": qu.sigmaz(),
                "sm": qu.sigmam(),
            },
            "cavity": {"a": qu.destroy(N_c)},
        },
        "state_params": {"l": l, "delta": delta},
        "timings": {"t_B": np.linspace(0, 1.0, 51),},
        "c_ops": [
            qu.tensor(qu.sigmam(), qu.qeye(N_c)) / np.sqrt(T_1q),
            qu.tensor(qu.qeye(N_q), qu.destroy(N_c)) / np.sqrt(T_1c),
            # qu.tensor(qu.qeye(N_q) - qu.sigmaz(), qu.qeye(N_c)) / np.sqrt(2 * T_qphi),
            qu.tensor(qu.sigmaz(), qu.qeye(N_c)) / np.sqrt(2 * T_qphi),
            qu.tensor(qu.qeye(N_q), np.sqrt(2) * qu.destroy(N_c).dag() * qu.destroy(N_c)) / np.sqrt(T_cphi),
        ],
        "simulations": {
            "decay_ratio": params["decays"]["ratio"],
            "r": r,
            "M": M,
            "sensor": init_state,
            # "sensor": qu.fock_dm(N_c, 0), #coherent state
            'p_0': l * params['simulations']['p_0'],
        },
    }

    # defining the displacement list
    vlist = np.linspace(
        -params["simulations"]["v_lim"] * l,
        params["simulations"]["v_lim"] * l,
        params["simulations"]["n_v"],
    )

    p_array = np.empty((len(vlist), 2 ** (r)))
    rho_f_array = np.empty((len(vlist), N_c, N_c), dtype=complex)

    
    # runing simulations
    p_array = np.array(p_array)
    parallel = Parallel(n_jobs=params["simulations"]["n_jobs"])
    output_generator = parallel(delayed(sBs_stabilization_autonomous)(v, system_dict) for v in tqdm(vlist))

    for idx, v in enumerate(vlist):
        rho_f_array[idx, :, :] = output_generator[idx]

    p_array = np.array(p_array)
    variance_list = np.linspace(0.01, l / 4, 101)
    MSD_array = np.empty((len(variance_list), len(vlist)))
    MSD_array_displacement = np.empty((len(variance_list), len(vlist)))
    posterior_array = np.empty((len(variance_list), 2 ** (r), len(vlist)))

    save_plot_data(
        param_filename,
        MSD_array,
        MSD_array_displacement,
        p_array,
        rho_f_array,
        vlist,
        variance_list,
        system_dict,
        results_options,
    )
    return None

def run_simulations_autonomous_eachnoise(param_filename, param_path, results_options):
    ''' only changed is how delta is read from the params file and how collapse operators are defined'''
    params = yaml.safe_load(Path(param_path + param_filename).read_text())
    r = params["simulations"]["r"]
    M = params["simulations"]["M"]
    # delta = params["state_params"] ["delta"]
    delta = results_options["Delta"]
    l = eval(params["state_params"]["l"])
    t_cd_B = params["simulations"]["t_CD_B"]

    N_c = params["dimensions"]["cavity"]
    N_q = params["dimensions"]["qubit"]
    qubit_T1 = params["decays"]["qubit_T1"]
    qubit_T2 = params["decays"]["qubit_T2"]
    cavity_T1 = params["decays"]["cavity_T1"]
    cavity_T2 = params["decays"]["cavity_T2"]
    
    T_1q = round(qubit_T1/ t_cd_B, 5)
    T_2q = round(qubit_T2/ t_cd_B, 5)
    T_1c = round(cavity_T1/ t_cd_B, 5)
    T_2c = round(cavity_T2/ t_cd_B, 5)

    T_qphi = 1 / (1 / T_2q - 1 / (2 * T_1q))
    T_cphi = 1 / (1 / T_2c - 1 / (2 * T_1c))

    if results_options["coherent"]:
        init_state = qu.fock_dm(N_c, 0)
        print('initial state coherent')
    else:
        print('initial state stabilized sensor')
        init_state_dict = np.load(
                "../Results/Input_states/coherent_sensor_state_delta_{}-Nc{}-{}.npz".format(int(100 * delta), N_c, params["decays"]["ratio"])
            )
        init_state = qu.Qobj(init_state_dict['rho_f_array'][0])
        

    #choosing collapse operators, given by file number
    file_number = results_options["file_number"]
    if file_number in results_options["collapse_dict"]["T1c"]:
        c_ops = [qu.tensor(qu.qeye(N_q), qu.destroy(N_c)) / np.sqrt(T_1c)]
        tag = "T1c"
    elif file_number in results_options["collapse_dict"]["T2c"]:
        c_ops = [qu.tensor(qu.qeye(N_q), np.sqrt(2) * qu.destroy(N_c).dag() * qu.destroy(N_c)) / np.sqrt(T_cphi)]
        tag = "T2c"
    elif file_number in results_options["collapse_dict"]["T1q"]:
        c_ops = [qu.tensor(qu.sigmam(), qu.qeye(N_c)) / np.sqrt(T_1q)]
        tag = "T1q"
    elif file_number in results_options["collapse_dict"]["T2q"]:
        c_ops = [qu.tensor(qu.sigmaz(), qu.qeye(N_c)) / np.sqrt(2 * T_qphi)]
        tag = "T2q"
    else:
        raise ValueError("File number does not match any collapse operator category.", file_number)

    system_dict = {
        "dimensions": {"qubit": N_q, "cavity": N_c,},
        "operators": {
            "qubit": {
                "sx": qu.sigmax(),
                "sy": qu.sigmay(),
                "sz": qu.sigmaz(),
                "sm": qu.sigmam(),
            },
            "cavity": {"a": qu.destroy(N_c)},
        },
        "state_params": {"l": l, "delta": delta},
        "timings": {"t_B": np.linspace(0, 1.0, 51),},
        "c_ops": c_ops,
        "simulations": {
            "decay_ratio": params["decays"]["ratio"],
            "r": r,
            "M": M,
            "sensor": init_state,
            # "sensor": qu.fock_dm(N_c, 0), #coherent state
            'p_0': l * params['simulations']['p_0'],
        },
    }

    # defining the displacement list
    vlist = np.linspace(
        -params["simulations"]["v_lim"] * l,
        params["simulations"]["v_lim"] * l,
        params["simulations"]["n_v"],
    )

    p_array = np.empty((len(vlist), 2 ** (r)))
    rho_f_array = np.empty((len(vlist), N_c, N_c), dtype=complex)

    
    # runing simulations
    p_array = np.array(p_array)
    parallel = Parallel(n_jobs=params["simulations"]["n_jobs"])
    output_generator = parallel(delayed(sBs_stabilization_autonomous)(v, system_dict) for v in tqdm(vlist))

    for idx, v in enumerate(vlist):
        rho_f_array[idx, :, :] = output_generator[idx]

    p_array = np.array(p_array)
    variance_list = np.linspace(0.01, l / 4, 101)
    MSD_array = np.empty((len(variance_list), len(vlist)))
    MSD_array_displacement = np.empty((len(variance_list), len(vlist)))
    posterior_array = np.empty((len(variance_list), 2 ** (r), len(vlist)))

    save_plot_data_eachnoise(
        param_filename,
        MSD_array,
        MSD_array_displacement,
        p_array,
        rho_f_array,
        vlist,
        variance_list,
        system_dict,
        results_options,
        tag,
    )
    return None



def run_simulations_backaction_withrhos(param_filename, param_path, results_options):
    '''
    function to run the backaction evading experiment
    saves the final density matrices
    see function backaction_evading in backaction_noise.py
    '''
    params = yaml.safe_load(Path(param_path + param_filename).read_text())
    delta = params["state_params"]["delta"]
    l = eval(params["state_params"]["l"])
    t_cd_B = params["simulations"]["t_CD_B"]

    N_c = params["dimensions"]["cavity"]
    N_q = params["dimensions"]["qubit"]
    qubit_T1 = params["decays"]["qubit_T1"]
    qubit_T2 = params["decays"]["qubit_T2"]
    cavity_T1 = params["decays"]["cavity_T1"]
    cavity_T2 = params["decays"]["cavity_T2"]
    
    T_1q = round(qubit_T1/ t_cd_B, 5)
    T_2q = round(qubit_T2/ t_cd_B, 5)
    T_1c = round(cavity_T1/ t_cd_B, 5)
    T_2c = round(cavity_T2/ t_cd_B, 5)

    T_qphi = 1 / (1 / T_2q - 1 / (2 * T_1q))
    T_cphi = 1 / (1 / T_2c - 1 / (2 * T_1c))

    if results_options["coherent"]:
        init_state = qu.fock_dm(N_c, 0)
        print('initial state coherent')
    else:
        print('initial state stabilized sensor')
        init_state_dict = np.load(
                "../Results/Input_states/coherent_sensor_state_delta_{}-Nc{}-{}.npz".format(int(100 * delta), N_c, params["decays"]["ratio"])
            )
        init_state = qu.Qobj(init_state_dict['rho_f_array'][0])
        

    system_dict = {
        "dimensions": {"qubit": N_q, "cavity": N_c,},
        "operators": {
            "qubit": {
                "sx": qu.sigmax(),
                "sy": qu.sigmay(),
                "sz": qu.sigmaz(),
                "sm": qu.sigmam(),
            },
            "cavity": {"a": qu.destroy(N_c)},
        },
        "state_params": {"l": l, "delta": delta},
        "timings": {"t_B": np.linspace(0, 1.0, 51),},
        "c_ops": [
            qu.tensor(qu.sigmam(), qu.qeye(N_c)) / np.sqrt(T_1q),
            qu.tensor(qu.qeye(N_q), qu.destroy(N_c)) / np.sqrt(T_1c),
            # qu.tensor(qu.qeye(N_q) - qu.sigmaz(), qu.qeye(N_c)) / np.sqrt(2 * T_qphi),
            qu.tensor(qu.sigmaz(), qu.qeye(N_c)) / np.sqrt(2 * T_qphi),
            qu.tensor(qu.qeye(N_q), np.sqrt(2) * qu.destroy(N_c).dag() * qu.destroy(N_c)) / np.sqrt(T_cphi),
        ],
        "simulations": {
            "decay_ratio": params["decays"]["ratio"],
            "T": params["simulations"]["T"],
            "M": params["simulations"]["M"],
            "N": params["simulations"]["N"],
            "v_lim": params["simulations"]["v_lim"],
            "n_v": params["simulations"]["n_v"],
            "sensor": init_state,
            "stddev": params["simulations"]["stddev"],
            # "sensor": qu.fock_dm(N_c, 0), #coherent state
        },
        "data": {
            "idx_data_q": params["data"]["idx_data_q"],
            "idx_data_p": params["data"]["idx_data_p"],
            "path_data": os.path.dirname(os.getcwd()) + "/Results/q_or_p_bitstring_paths/",
        },
    }

    output_generator = Parallel(n_jobs=params["simulations"]["n_jobs"])(delayed(backaction_evading)(system_dict) for i in tqdm(range(params["simulations"]["repeat"])))
    # output_generator = backaction_evading(system_dict)
    
    squared_errors_q, squared_errors_p = [], []
    estimators_q, estimators_p = [], []
    q0s, p0s = [], []
    rhos = []

    for idx in range(params["simulations"]["repeat"]):
        squared_errors_q.append(output_generator[idx][0])
        squared_errors_p.append(output_generator[idx][1])
        estimators_q.append(output_generator[idx][2])
        estimators_p.append(output_generator[idx][3])
        q0s.append(output_generator[idx][4])
        p0s.append(output_generator[idx][5])
        rhos.append(output_generator[idx][6])

    np.savez(
            results_options["save_path_name"] + param_filename[:-5],
            # results_options["save_path_name"] + str(system_dict["simulations"]["decay_ratio"]),
            system_dict=system_dict,
            rhos = rhos,
            squared_errors_q=squared_errors_q,
            squared_errors_p=squared_errors_p,
            estimators_q=estimators_q,
            estimators_p=estimators_p,
            q0s=q0s,
            p0s=p0s,
        )
    return None



def run_simulations_backaction_notrhos(param_filename, param_path, results_options):
    '''
    function to run the backaction evading experiment
    see function backaction_evading_notrhos in backaction_noise.py
    '''
    params = yaml.safe_load(Path(param_path + param_filename).read_text())
    delta = params["state_params"]["delta"]
    l = eval(params["state_params"]["l"])
    t_cd_B = params["simulations"]["t_CD_B"]

    N_c = params["dimensions"]["cavity"]
    N_q = params["dimensions"]["qubit"]
    qubit_T1 = params["decays"]["qubit_T1"]
    qubit_T2 = params["decays"]["qubit_T2"]
    cavity_T1 = params["decays"]["cavity_T1"]
    cavity_T2 = params["decays"]["cavity_T2"]
    
    T_1q = round(qubit_T1/ t_cd_B, 5)
    T_2q = round(qubit_T2/ t_cd_B, 5)
    T_1c = round(cavity_T1/ t_cd_B, 5)
    T_2c = round(cavity_T2/ t_cd_B, 5)

    T_qphi = 1 / (1 / T_2q - 1 / (2 * T_1q))
    T_cphi = 1 / (1 / T_2c - 1 / (2 * T_1c))

    if results_options["coherent"]:
        init_state = qu.fock_dm(N_c, 0)
        print('initial state coherent')
    else:
        print('initial state stabilized sensor')
        init_state_dict = np.load(
                "../Results/Input_states/coherent_sensor_state_delta_{}-Nc{}-{}.npz".format(int(100 * delta), N_c, params["decays"]["ratio"])
            )
        init_state = qu.Qobj(init_state_dict['rho_f_array'][0])
        

    system_dict = {
        "dimensions": {"qubit": N_q, "cavity": N_c,},
        "operators": {
            "qubit": {
                "sx": qu.sigmax(),
                "sy": qu.sigmay(),
                "sz": qu.sigmaz(),
                "sm": qu.sigmam(),
            },
            "cavity": {"a": qu.destroy(N_c)},
        },
        "state_params": {"l": l, "delta": delta},
        "timings": {"t_B": np.linspace(0, 1.0, 51),},
        "c_ops": [
            qu.tensor(qu.sigmam(), qu.qeye(N_c)) / np.sqrt(T_1q),
            qu.tensor(qu.qeye(N_q), qu.destroy(N_c)) / np.sqrt(T_1c),
            # qu.tensor(qu.qeye(N_q) - qu.sigmaz(), qu.qeye(N_c)) / np.sqrt(2 * T_qphi),
            qu.tensor(qu.sigmaz(), qu.qeye(N_c)) / np.sqrt(2 * T_qphi),
            qu.tensor(qu.qeye(N_q), np.sqrt(2) * qu.destroy(N_c).dag() * qu.destroy(N_c)) / np.sqrt(T_cphi),
        ],
        "simulations": {
            "decay_ratio": params["decays"]["ratio"],
            "T": params["simulations"]["T"],
            "M": params["simulations"]["M"],
            "N": params["simulations"]["N"],
            "v_lim": params["simulations"]["v_lim"],
            "n_v": params["simulations"]["n_v"],
            "sensor": init_state,
            "stddev": params["simulations"]["stddev"],
            # "sensor": qu.fock_dm(N_c, 0), #coherent state
        },
        "data": {
            "idx_data_q": params["data"]["idx_data_q"],
            "idx_data_p": params["data"]["idx_data_p"],
            "path_data": os.path.dirname(os.getcwd()) + "/Results/q_or_p_bitstring_paths/",
        },
    }

    output_generator = Parallel(n_jobs=params["simulations"]["n_jobs"])(delayed(backaction_evading_notrhos)(system_dict) for i in tqdm(range(params["simulations"]["repeat"])))
    # output_generator = backaction_evading_notrhos(system_dict)

    squared_errors_q, squared_errors_p = [], []
    estimators_q, estimators_p = [], []
    q0s, p0s = [], []
    bitstrings = []

    for idx in range(params["simulations"]["repeat"]):
        squared_errors_q.append(output_generator[idx][0])
        squared_errors_p.append(output_generator[idx][1])
        estimators_q.append(output_generator[idx][2])
        estimators_p.append(output_generator[idx][3])
        q0s.append(output_generator[idx][4])
        p0s.append(output_generator[idx][5])
        bitstrings.append(output_generator[idx][6])

    np.savez(
            results_options["save_path_name"] + param_filename[:-5],
            # results_options["save_path_name"] + str(system_dict["simulations"]["decay_ratio"]),
            system_dict=system_dict,
            squared_errors_q=squared_errors_q,
            squared_errors_p=squared_errors_p,
            estimators_q=estimators_q,
            estimators_p=estimators_p,
            q0s=q0s,
            p0s=p0s,
            bitstrings=bitstrings,
        )
    return None