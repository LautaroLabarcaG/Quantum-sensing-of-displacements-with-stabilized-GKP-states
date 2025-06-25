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

def FI_analytical(q,Delta):
    '''
    FI of one bit. 
    '''
    cd = np.cosh(Delta**2)
    a = 0.4
    numerator =  np.exp(-2*a*Delta**2)*l**2*cd**2*np.cos(l*cd*q)**2 
    denominator = 1-np.sin(l*cd*q)**2*np.exp(-2*a*Delta**2)
    return numerator/denominator

def FI_qq_T(probs_qp0,R1,q0range,T,steps):
    '''
    probs_qp0: array with the probability of measuring the bit at g in round R, up to round R1. Shape [q0][q or p][R]

    returns the Fisher information respect q, as a function of q0 and p0 fixed. shape [FI_q(q0)]. 
    '''

    probs_qp0_allbits = []
    for index_q0 in range(steps):
        probs_q0p0 = probs_qp0[index_q0]
        probs_q0p0_allbits = get_allqbits_q0p0_T(probs_q0p0,R1,T)
        probs_qp0_allbits.append(probs_q0p0_allbits)
    probs_qp0_allbits = np.array(probs_qp0_allbits)
    
    probs_allbits_qp0 = np.transpose(probs_qp0_allbits)#these are the likelihoods
    partialq_allbits_qp0 = np.gradient(probs_allbits_qp0, q0range, axis=1)

    def FI(probs_qp0_allbits,partialq_allbits_qp0,index_q0):
        probs_q0p0_allbits = probs_qp0_allbits[index_q0]
        FI_qq_q0p0 = 0
        for b in range(2**T):
            prob = probs_q0p0_allbits[b]
            partialq_b_q0 = partialq_allbits_qp0[b][index_q0]
            if prob>0:
                FI_qq_q0p0 += (partialq_b_q0)**2/prob
            else:
                FI_qq_q0p0 += 0
        return FI_qq_q0p0

    FIqq = np.zeros(len(q0range))
    for index_q0 in range(len(q0range)):
        FIqq[index_q0] = FI(probs_qp0_allbits,partialq_allbits_qp0,index_q0)
    return FIqq


def FI_qq_T_fullmap(probs_qp0_allbits,q0range,R,T,steps):
    '''
    probs_qp0: array with the probabilities of each bitstring of length 2*R, with shape [q0][2**(2*R)]

    returns the Fisher information respect q, as a function of q0 and p0 fixed. shape [FI_q(q0)]. 
    '''
    
    probs_qp0_Tbits = []
    for index_q0 in range(steps):
        probs_q0p0 = probs_qp0_allbits[index_q0]
        probs_Tbits_q0p0 = get_probabilities_bitsT(probs_q0p0,R,T)
        probs_qp0_Tbits.append(probs_Tbits_q0p0)
    probs_qp0_Tbits = np.array(probs_qp0_Tbits)
    probs_Tbits_qp0 = np.transpose(probs_qp0_Tbits)#these are the likelihoods
    partialq_Tbits_qp0 = np.gradient(probs_Tbits_qp0, q0range, axis=1)

    def FI(probs_qp0_allbits,partialq_allbits_qp0,index_q0):
        probs_q0p0_allbits = probs_qp0_allbits[index_q0]
        FI_qq_q0p0 = 0
        for b in range(2**(2*T)):
            prob = probs_q0p0_allbits[b]
            partialq_b_q0 = partialq_allbits_qp0[b][index_q0]
            if prob>0:
                FI_qq_q0p0 += (partialq_b_q0)**2/prob
            else:
                FI_qq_q0p0 += 0
        return FI_qq_q0p0
        
    FIqq = np.zeros(len(q0range))
    for index_q0 in range(len(q0range)):
        FIqq[index_q0] = FI(probs_qp0_Tbits,partialq_Tbits_qp0,index_q0)
    return FIqq

def FI_qq_T_qmap(probs_qp0_qbits,q0range,R,T,steps):
    '''
    probs_qp0_qbits: array with the probabilities of all q bitstrings, up to round R1. Shape [q0][2**(T)].

    returns the Fisher information respect q, as a function of q0 and p0 fixed. shape [FI_q(q0)]. 
    '''
    
    probs_qp0_Tbits = []
    for index_q0 in range(steps):
        probs_q0p0 = probs_qp0_qbits[index_q0]
        probs_Tbits_q0p0 = get_probabilities_qbitsT(probs_q0p0,R,T)
        probs_qp0_Tbits.append(probs_Tbits_q0p0)
    probs_qp0_Tbits = np.array(probs_qp0_Tbits)
    probs_Tbits_qp0 = np.transpose(probs_qp0_Tbits)#these are the likelihoods
    partialq_Tbits_qp0 = np.gradient(probs_Tbits_qp0, q0range, axis=1)

    def FI(probs_qp0_allbits,partialq_allbits_qp0,index_q0):
        probs_q0p0_allbits = probs_qp0_allbits[index_q0]
        FI_qq_q0p0 = 0
        for b in range(2**(T)):
            prob = probs_q0p0_allbits[b]
            partialq_b_q0 = partialq_allbits_qp0[b][index_q0]
            if prob>0:
                FI_qq_q0p0 += (partialq_b_q0)**2/prob
            else:
                FI_qq_q0p0 += 0
        return FI_qq_q0p0
        
    FIqq = np.zeros(len(q0range))
    for index_q0 in range(len(q0range)):
        FIqq[index_q0] = FI(probs_qp0_Tbits,partialq_Tbits_qp0,index_q0)
    return FIqq

def ML_q0(probs_qp0,R1,q0range,T,steps):
    '''
    probs_qp0: array with the probability of measuring the bit at g in round R, up to round R1. Shape [q0][q or p][R]

    returns the ML estimator of q0, as a function of q0 and p0 fixed.
    '''

    probs_allbits_qp0 = []
    for index_q0 in range(steps):
        probs_q0p0 = probs_qp0[index_q0]
        probs_allbits_q0p0 = get_allqbits_q0p0_T(probs_q0p0,R1,T)
        probs_allbits_qp0.append(probs_allbits_q0p0)
    
    probs_allbits_qp0 = np.transpose(np.array(probs_allbits_qp0))#these are the likelihoods
    ML_index_q0 = np.argmax(probs_allbits_qp0,axis=1)
    MLq = q0range[ML_index_q0]
    return MLq

def ML_p0(probs_pq0,R1,p0range,T,steps):
    '''
    probs_qp0: array with the probability of measuring the bit at g in round R, up to round R1. Shape [p0][q or p][R]

    returns the ML estimator of p0, as a function of p0 and q0 fixed.
    '''

    probs_allbits_pq0 = []
    for index_p0 in range(steps):
        probs_q0p0 = probs_pq0[index_p0]
        probs_allbits_q0p0 = get_allpbits_q0p0_T(probs_q0p0,R1,T)
        probs_allbits_pq0.append(probs_allbits_q0p0)
    
    probs_allbits_pq0 = np.transpose(np.array(probs_allbits_pq0))#these are the likelihoods
    ML_index_p0 = np.argmax(probs_allbits_pq0,axis=1)
    MLp = p0range[ML_index_p0]
    return MLp

def ML_q0_fullmap(probs_qp0_allbits,R1,q0range,T,steps):
    '''
    probs_qp0: array with the probability of measuring the bit at g in round R, up to round R1. Shape [q0][q or p][R], i.e. p(bit|q0,p0)

    returns the average mean square error of the ML estimator of q0, as a function of q0 and p0 fixed.
    '''

    probs_qp0_Tbits = []
    for index_q0 in range(steps):
        probs_q0p0 = probs_qp0_allbits[index_q0]
        probs_Tbits_q0p0 = get_probabilities_bitsT(probs_q0p0,R1,T)
        probs_qp0_Tbits.append(probs_Tbits_q0p0)
    probs_qp0_Tbits = np.array(probs_qp0_Tbits)
    probs_Tbits_qp0 = np.transpose(probs_qp0_Tbits)#these are the likelihoods p(q0,p0|b)
    partialq_Tbits_qp0 = np.gradient(probs_Tbits_qp0, q0range, axis=1)
    
    ML_index_q0 = np.argmax(probs_Tbits_qp0,axis=1)
    ML_q0 = q0range[ML_index_q0]
    return ML_q0

def ML_q0_full_q_map(probs_qp0_qbits,R1,q0range,T,steps):
    '''
    probs_qp0_Tbits: array with the probabilities of all q bitstrings, up to round R1. Shape [q0][2**(T)].

    returns the average mean square error of the ML estimator of q0, as a function of q0 and p0 fixed.
    '''

    probs_qp0_Tbits = []
    for index_q0 in range(steps):
        probs_q0p0 = probs_qp0_qbits[index_q0]
        probs_Tbits_q0p0 = get_probabilities_qbitsT(probs_q0p0,R1,T)
        probs_qp0_Tbits.append(probs_Tbits_q0p0)
    probs_qp0_Tbits = np.array(probs_qp0_Tbits)
    probs_Tbits_qp0 = np.transpose(probs_qp0_Tbits)#these are the likelihoods
    partialq_Tbits_qp0 = np.gradient(probs_Tbits_qp0, q0range, axis=1)
    
    ML_index_q0 = np.argmax(probs_Tbits_qp0,axis=1)
    ML_q0 = q0range[ML_index_q0]
    return ML_q0


def MSE_ML_q0(probs_qp0,R1,q0range,T,steps):
    '''
    probs_qp0: array with the probability of measuring the bit at g in round R, up to round R1. Shape [q0][q or p][R]

    returns the average mean square error of the ML estimator of q0, as a function of q0 and p0 fixed.
    '''

    probs_allbits_qp0 = []
    for index_q0 in range(steps):
        probs_q0p0 = probs_qp0[index_q0]
        probs_allbits_q0p0 = get_allqbits_q0p0_T(probs_q0p0,R1,T)
        probs_allbits_qp0.append(probs_allbits_q0p0)
    
    probs_allbits_qp0 = np.transpose(np.array(probs_allbits_qp0))#these are the likelihoods
    ML_index_q0 = np.argmax(probs_allbits_qp0,axis=1)
    ML_q0 = q0range[ML_index_q0]

    MSE, means, biases, Var = [], [],[],[]
    for index_q0 in range(steps):
        q0 = q0range[index_q0]
        MSE_q0 = 0
        mean = 0
        Var_q0 = 0
        for b in range(2**T):
            prob_b_q0 = probs_allbits_qp0[b][index_q0]
            estimator_q0_b = ML_q0[b]
            mean += prob_b_q0*estimator_q0_b
            MSE_q0 += prob_b_q0*(estimator_q0_b-q0)**2
        MSE.append(MSE_q0)
        means.append(mean)
        biases.append(mean-q0)
        for b in range(2**T):
            prob_b_q0 = probs_allbits_qp0[b][index_q0]
            Var_q0 += prob_b_q0*(ML_q0[b]-mean)**2
        Var.append(Var_q0)
    partial_means = np.gradient(means,q0range)
    return np.array(MSE), np.array(biases), partial_means, np.array(Var)

def MSE_ML_q0_fullmap(probs_qp0_allbits,R1,q0range,T,steps):
    '''
    probs_qp0: array with the probability of measuring the bit at g in round R, up to round R1. Shape [q0][q or p][R]

    returns the average mean square error of the ML estimator of q0, as a function of q0 and p0 fixed.
    '''

    probs_qp0_Tbits = []
    for index_q0 in range(steps):
        probs_q0p0 = probs_qp0_allbits[index_q0]
        probs_Tbits_q0p0 = get_probabilities_bitsT(probs_q0p0,R1,T)
        probs_qp0_Tbits.append(probs_Tbits_q0p0)
    probs_qp0_Tbits = np.array(probs_qp0_Tbits)
    probs_Tbits_qp0 = np.transpose(probs_qp0_Tbits)#these are the likelihoods
    partialq_Tbits_qp0 = np.gradient(probs_Tbits_qp0, q0range, axis=1)
    
    ML_index_q0 = np.argmax(probs_Tbits_qp0,axis=1)
    ML_q0 = q0range[ML_index_q0]

    MSE, means, biases, Var = [], [],[],[]
    for index_q0 in range(steps):
        q0 = q0range[index_q0]
        MSE_q0 = 0
        mean = 0
        Var_q0 = 0
        for b in range(2**(2*T)):
            prob_b_q0 = probs_Tbits_qp0[b][index_q0]
            estimator_q0_b = ML_q0[b]
            mean += prob_b_q0*estimator_q0_b
            MSE_q0 += prob_b_q0*(estimator_q0_b-q0)**2
        MSE.append(MSE_q0)
        means.append(mean)
        biases.append(mean-q0)
        for b in range(2**(2*T)):
            prob_b_q0 = probs_Tbits_qp0[b][index_q0]
            Var_q0 += prob_b_q0*(ML_q0[b]-mean)**2
        Var.append(Var_q0)
    partial_means = np.gradient(means,q0range)
    return np.array(MSE), np.array(biases), partial_means, np.array(Var)

def MSE_ML_q0_q_map(probs_qp0_qbits,R1,q0range,T,steps):
    '''
    probs_qp0_Tbits: array with the probabilities of all q bitstrings, up to round R1. Shape [q0][2**(T)].

    returns the average mean square error of the ML estimator of q0, as a function of q0 and p0 fixed.
    '''

    probs_qp0_Tbits = []
    for index_q0 in range(steps):
        probs_q0p0 = probs_qp0_qbits[index_q0]
        probs_Tbits_q0p0 = get_probabilities_qbitsT(probs_q0p0,R1,T)
        probs_qp0_Tbits.append(probs_Tbits_q0p0)
    probs_qp0_Tbits = np.array(probs_qp0_Tbits)
    probs_Tbits_qp0 = np.transpose(probs_qp0_Tbits)#these are the likelihoods
    partialq_Tbits_qp0 = np.gradient(probs_Tbits_qp0, q0range, axis=1)
    
    ML_index_q0 = np.argmax(probs_Tbits_qp0,axis=1)
    ML_q0 = q0range[ML_index_q0]

    MSE, means, biases, Var = [], [],[],[]
    for index_q0 in range(steps):
        q0 = q0range[index_q0]
        MSE_q0 = 0
        mean = 0
        Var_q0 = 0
        for b in range(2**(T)):
            prob_b_q0 = probs_Tbits_qp0[b][index_q0]
            estimator_q0_b = ML_q0[b]
            mean += prob_b_q0*estimator_q0_b
            MSE_q0 += prob_b_q0*(estimator_q0_b-q0)**2
        MSE.append(MSE_q0)
        means.append(mean)
        biases.append(mean-q0)
        for b in range(2**(T)):
            prob_b_q0 = probs_Tbits_qp0[b][index_q0]
            Var_q0 += prob_b_q0*(ML_q0[b]-mean)**2
        Var.append(Var_q0)
    partial_means = np.gradient(means,q0range)
    return np.array(MSE), np.array(biases), partial_means, np.array(Var)

def sBs_random_bits_probs_estimators(Delta,q0,p0,sensor,T,estimators_allqbits_p0,estimators_allpbits_q0,gauges,exp_qrange_allqbits,qrange,steps):
    '''
        Delta:envelope variance
        q0,p0: initial displacement
        sensor: initial state
        T: number of q-p rounds used for estimation
        Krauss_dictionary: dictionary with the Krauss operators. These are used to get the probabilities.
        Krauss_dictionary_U: dictionary with the Krauss operators and feedback. These are used to update the state.
        estimators_allbits_q0p0: array with the estimators of q0 and p0 for each bitstring of length T. Shape [2**(T)]. These estimators are obtained with the averaged probabilities. 
        gauges: array with the initial gauges. 
        exp_qrange_allqbits: expected value of q after a displacement q0 in qrange, p0=l/4, and recovery bitstring b. Shape [q0][2**T].
        qrange: the range of exp_qrange_allqbits.
        steps: the steps in qrange. 

        returns rho,final_gauge, bits, estimatorq, estimatorp.
    ''' 

    beta = (q0 + 1j*p0)/np.sqrt(2)
    U_q0p0 = Displacement(beta)
    rho0 = U_q0p0@sensor@U_q0p0.getH()
    Krauss_dictionary, Krauss_dictionary_U = Krauss_dictionaries(Delta)
    rho, bitstring, probs_q, probs_p, final_gauge = sBs_random_run(Delta,rho0,T, Krauss_dictionary, Krauss_dictionary_U,gauges)
    bq, bp = bitstring[::2], bitstring[1::2]
    bq = binary_array_to_int(bq)
    bp = binary_array_to_int(bp)

    estimatorq = estimators_allqbits_p0[bq]
    estimatorp = estimators_allpbits_q0[bp]

    index_q0 = np.argmin(np.abs(qrange-estimatorq))
    index_p0 = np.argmin(np.abs(qrange-estimatorp))
    estq_final = exp_qrange_allqbits[index_q0][bq]
    estp_final = exp_qrange_allqbits[index_p0][bp]

    return rho,final_gauge, bitstring, estimatorq, estimatorp, estq_final, estp_final


#################################################################################
# gaussian prior
#################################################################################

def posterior_qp0_allbits(prior_q,probs_qp0_allbits,R,T,q0_range,steps):
    '''
    prior_q: shape [q0], prior p(q0)
    probs_qp0_allbits: shape [q0][bit] [steps][2**(2R)], p(b|q0,p0)
    R: length of the bits
    T: length of the bits to be used for the estimation
    q0_range: values of q0
    steps: number of values of q0

    returns the posterior probabilities of q0 and p0 for each bit
    shape [bit][q0], [2**(2T)][steps] p(q0,p0|b)
    and the likelihoods p(b|q0,p0) shape [steps][2**(2T)]
    '''
    probs_qp0_Tbits = []
    for index_q0 in range(steps):
        probs_q0p0 = probs_qp0_allbits[index_q0]
        probs_Tbits_q0p0 = get_probabilities_bitsT(probs_q0p0,R,T)
        probs_qp0_Tbits.append(probs_Tbits_q0p0)
    probs_qp0_Tbits = np.array(probs_qp0_Tbits)
    probs_Tbits_qp0 = np.transpose(probs_qp0_Tbits)#these are the likelihoods p(b|q0,p0)

    posteriors_qp0_b = []
    for b in range(2**(2*T)):
        p_bqp0 = probs_Tbits_qp0[b]#likelihood p(b|q0,p0), shape [steps]
        p_qp0b = p_bqp0*prior_q #posterior p(q0,p0|b), shape [steps]
        posteriors_qp0_b.append(p_qp0b/np.sum(p_qp0b))
    posteriors_qp0_b = np.array(posteriors_qp0_b)
    return posteriors_qp0_b, probs_Tbits_qp0

def posterior_qp0_qbits(prior_q,probs_qp0_allbits,R,T,q0_range,steps):
    '''
    prior_q: shape [q0], prior p(q0)
    probs_qp0_allbits: shape [q0][bit] [steps][2**(2R)], p(b|q0,p0)
    R: length of the bits
    T: length of the bits to be used for the estimation
    q0_range: values of q0
    steps: number of values of q0

    returns the posterior probabilities of q0 and p0 for each bit
    shape [bit][q0], [2**(2T)][steps] p(q0,p0|b)
    and the likelihoods p(b|q0,p0) shape [steps][2**(2T)]
    '''
    probs_qp0_Tbits = []
    for index_q0 in range(steps):
        probs_q0p0 = probs_qp0_allbits[index_q0]
        probs_Tbits_q0p0 = get_probabilities_qbitsT(probs_q0p0,R,T)
        probs_qp0_Tbits.append(probs_Tbits_q0p0)
    probs_qp0_Tbits = np.array(probs_qp0_Tbits)
    probs_Tbits_qp0 = np.transpose(probs_qp0_Tbits)#these are the likelihoods p(b|q0,p0)

    posteriors_qp0_b = []
    for b in range(2**(T)):
        p_bqp0 = probs_Tbits_qp0[b]#likelihood p(b|q0,p0), shape [steps]
        p_qp0b = p_bqp0*prior_q #posterior p(q0,p0|b), shape [steps]
        posteriors_qp0_b.append(p_qp0b/np.sum(p_qp0b))
    posteriors_qp0_b = np.array(posteriors_qp0_b)
    return posteriors_qp0_b, probs_Tbits_qp0

def gaussian_MSE(stdev,probs_qp0_allbits,R,T,q0_range,steps):
    '''
    stdev: standard deviation of the prior gaussian
    prior_q: shape [q0], prior p(q0)
    probs_qp0_allbits: shape [q0][bit] [steps][2**(2R)], p(b|q0,p0)
    R: length of the bits
    T: length of the bits to be used for the estimation
    q0_range: values of q0
    steps: number of values of q0
    '''

    prior_q = prior_gaussian(q0_range,stdev) #p(q0), gaussian, shape [steps]
    posterior_qp0, likelihoods = posterior_qp0_allbits(prior_q,probs_qp0_allbits,R,T,q0_range,steps)
    #posterior p(q0,p0|b), shape [bit][q0], [2**(2T)][steps]
    #likelihoods p(b|q0,p0), shape [q0][bit], [steps][2**(2T)]

    estimators = []
    for b in range(2**(2*T)):
        p_qp0b = posterior_qp0[b]
        q_b = np.sum(q0_range*p_qp0b)
        estimators.append(q_b)
        # print(int_to_r_bits(b,2*T),q_b)
    
    probs_b = []
    for b in range(2**(2*T)):
        likelihood = likelihoods[b]
        prob_b = np.sum(prior_q*likelihood)
        probs_b.append(prob_b)
    probs_b = np.array(probs_b)

    mean_squared_errors = []
    for q0_index in range(steps):
        q0 = q0_range[q0_index]
        prior_q0 = prior_q[q0_index]
        mse_q0 = 0
        for b in range(2**(2*T)):
            likelihood = likelihoods[b][q0_index]
            mse_q0+= prior_q0*likelihood*(q0-estimators[b])**2
        mean_squared_errors.append(mse_q0)
    mean_squared_errors = np.array(mean_squared_errors)
    MSE = np.sum(mean_squared_errors)
    return mean_squared_errors, MSE, posterior_qp0, probs_b

def gaussian_MSE_qmap(stdev,probs_qp0_allbits,R,T,q0_range,steps):
    '''
    stdev: standard deviation of the prior gaussian
    probs_qp0_allbits: shape [q0][bit] [steps][2**(R)], p(b|q0,p0)
    R: length of the bits
    T: length of the bits to be used for the estimation
    q0_range: values of q0
    steps: number of values of q0
    '''

    prior_q = prior_gaussian(q0_range,stdev) #p(q0), gaussian, shape [steps]
    posterior_qp0, likelihoods = posterior_qp0_qbits(prior_q,probs_qp0_allbits,R,T,q0_range,steps)
    #posterior p(q0,p0|b), shape [bit][q0], [2**(2T)][steps]
    #likelihoods p(b|q0,p0), shape [q0][bit], [steps][2**(2T)]

    estimators = []
    for b in range(2**(T)):
        p_qp0b = posterior_qp0[b]
        q_b = np.sum(q0_range*p_qp0b)
        estimators.append(q_b)
        # print(int_to_r_bits(b,2*T),q_b)
    
    probs_b = []
    for b in range(2**(T)):
        likelihood = likelihoods[b]
        prob_b = np.sum(prior_q*likelihood)
        probs_b.append(prob_b)
    probs_b = np.array(probs_b)

    mean_squared_errors = []
    for q0_index in range(steps):
        q0 = q0_range[q0_index]
        prior_q0 = prior_q[q0_index]
        mse_q0 = 0
        for b in range(2**(T)):
            likelihood = likelihoods[b][q0_index]
            mse_q0+= prior_q0*likelihood*(q0-estimators[b])**2
        mean_squared_errors.append(mse_q0)
    mean_squared_errors = np.array(mean_squared_errors)
    MSE = np.sum(mean_squared_errors)
    return mean_squared_errors, MSE, posterior_qp0, estimators, probs_b

def gaussian_MSEsquared_qmap(stdev,probs_qp0_allbits,R,T,q0_range,steps):
    '''
    exactly the same as gaussian_MSE_qmap, but with the MSE squared.
    '''

    prior_q = prior_gaussian(q0_range,stdev) #p(q0), gaussian, shape [steps]
    posterior_qp0, likelihoods = posterior_qp0_qbits(prior_q,probs_qp0_allbits,R,T,q0_range,steps)
    #posterior p(q0,p0|b), shape [bit][q0], [2**(2T)][steps]
    #likelihoods p(b|q0,p0), shape [q0][bit], [steps][2**(2T)]

    estimators = []
    for b in range(2**(T)):
        p_qp0b = posterior_qp0[b]
        q_b = np.sum(q0_range*p_qp0b)
        estimators.append(q_b)
        # print(int_to_r_bits(b,2*T),q_b)
    
    probs_b = []
    for b in range(2**(T)):
        likelihood = likelihoods[b]
        prob_b = np.sum(prior_q*likelihood)
        probs_b.append(prob_b)
    probs_b = np.array(probs_b)

    mean_squared_errors = []
    for q0_index in range(steps):
        q0 = q0_range[q0_index]
        prior_q0 = prior_q[q0_index]
        mse_q0 = 0
        for b in range(2**(T)):
            likelihood = likelihoods[b][q0_index]
            mse_q0+= prior_q0*likelihood*(q0-estimators[b])**4
        mean_squared_errors.append(mse_q0)
    mean_squared_errors = np.array(mean_squared_errors)
    MSE = np.sum(mean_squared_errors)
    return mean_squared_errors, MSE, posterior_qp0, estimators, probs_b


def prior_gaussian(q0_range,stddev):
    '''
    q0_range: values of q0
    variance: variance of the gaussian
    '''
    prior_q = np.exp(-q0_range**2/2/stddev**2)
    prior_q = prior_q/np.sum(prior_q)
    return prior_q

def flat_MSE_qmap(probs_qp0_allbits,R,T,q0_range,steps):
    '''
    probs_qp0_allbits: shape [q0][bit] [steps][2**(R)], p(b|q0,p0)
    R: length of the bits
    T: length of the bits to be used for the estimation
    q0_range: values of q0
    steps: number of values of q0
    '''

    prior_q = np.ones(steps)/steps #p(q0), flat, shape [steps]
    posterior_qp0, likelihoods = posterior_qp0_qbits(prior_q,probs_qp0_allbits,R,T,q0_range,steps)
    #posterior p(q0,p0|b), shape [bit][q0], [2**(2T)][steps]
    #likelihoods p(b|q0,p0), shape [q0][bit], [steps][2**(2T)]

    estimators = []
    for b in range(2**(T)):
        q_b = np.sum(q0_range*posterior_qp0[b])
        estimators.append(q_b)
        # print(int_to_r_bits(b,2*T),q_b)

    probs_b = []
    for b in range(2**(T)):
        likelihood = likelihoods[b]
        prob_b = np.sum(prior_q*likelihood)
        probs_b.append(prob_b)
    probs_b = np.array(probs_b)

    mean_squared_errors = []
    for q0_index in range(steps):
        q0 = q0_range[q0_index]
        prior_q0 = prior_q[q0_index]
        mse_q0 = 0
        for b in range(2**(T)):
            likelihood = likelihoods[b][q0_index]
            mse_q0+= prior_q0*likelihood*(q0-estimators[b])**2
        mean_squared_errors.append(mse_q0)
    mean_squared_errors = np.array(mean_squared_errors)
    MSE = np.sum(mean_squared_errors)
    return mean_squared_errors, MSE, posterior_qp0, estimators, probs_b