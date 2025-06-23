import numpy as np
import qutip as qt
import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
from tqdm import tqdm
from joblib import Parallel, delayed
import os

def int_to_r_bits(n, r):
    """
    Converts an integer to its r-bits binary representation.
    """
    if n < 0 or n >= 2**r:
        raise ValueError("n must be non-negative and less than 2^r.")
    return format(n, f'0{r}b')

def binary_array_to_int(binary_array):
    """
    Convert an array of 0s and 1s to the corresponding integer.
    
    Parameters:
    binary_array (array-like): Array of 0s and 1s.
    
    Returns:
    int: The corresponding integer.
    """
    binary_string = ''.join(str(bit) for bit in binary_array)
    return int(binary_string, 2)

def get_probability_bp(probs_bitstrings_p0q0,T):
    '''
    probs_bitstrings_pq: array with the probability of all bitstrings with length 2T, that is both q and p bits, for a given displacement q0,p0. 
    T: number of q-p cycles

    Returns the probability of all the bitstrings bp, of length T. 
    '''

    probs_bp = []
    for b in range(2**T):
        bp = int_to_r_bits(b,T)
        prob_bp = 0
        for i in range(2**(2*T)):
            bqp = int_to_r_bits(i,2*T)
            if bp == bqp[1::2]:
                prob_bp += probs_bitstrings_p0q0[i]
        probs_bp.append(prob_bp)
    return np.array(probs_bp)

def get_probability_bq(probs_bitstrings_p0q0,T):
        '''
        probs_bitstrings_p0q0: array with the probability of all bitstrings with length 2T, that is both q and p bits, for a given displacement q0,p0. 
        T: number of q-p cycles

        Returns the probability of all the bitstrings bq, and bp, each of length T. 
        '''

        probs_bq = []
        for b in range(2**T):
            bq = int_to_r_bits(b,T)
            prob_bq = 0
            for i in range(2**(2*T)):
                bqp = int_to_r_bits(i,2*T)
                if bq == bqp[::2]:
                    prob_bq += probs_bitstrings_p0q0[i]
            probs_bq.append(prob_bq)
        return np.array(probs_bq)

def get_probability_bq_bp(probs_allbits,T):
    '''
    probs_bitstrings_pq: array with the probability of all bitstrings with length 2T, that is both q and p bits, for a given displacement q0,p0. 
    T: number of q-p cycles

    Returns the probability of all the bitstrings bp,bq of length T. 
    '''

    probs_q, probs_p = np.zeros(2**T), np.zeros(2**T)
    for j in range(2**T):
        bitqp = int_to_r_bits(j,T)
        prob_q, prob_p = 0,0
        for b in range(2**(2*T)):
            bit = int_to_r_bits(b,2*T)
            bq, bp = bit[::2], bit[1::2]
            if bitqp == bq:
                prob_q += probs_allbits[b]
            if bitqp == bp:
                prob_p += probs_allbits[b]
        probs_q[j] = prob_q
        probs_p[j] = prob_p
        # print(bitqp,np.round(prob_q,2),np.round(prob_p,2))
    return probs_q, probs_p

def get_probability_of_each_bit(probs_allbits,T):
    '''
    probs_bitstrings_pq: array with the probability of all bitstrings with length 2T, that is both q and p bits, for a given displacement q0,p0. 
    T: number of q-p cycles

    Returns the probability of each bit being 0, for q and p, up to T cycles.
    '''

    probs_q, probs_p = get_probability_bq_bp(probs_allbits,T)
    prob_each_q, prob_each_p = np.zeros(T), np.zeros(T)
    for j in range(T):
        for b in range(2**T):
            bit = int_to_r_bits(b,T)
            if bit[j]=='0':
                prob_each_q[j] += probs_q[b]
                prob_each_p[j] += probs_p[b]
        # print('prob of 0 at',j,np.round(prob_each_q[j],2),np.round(prob_each_p[j],2))
    return prob_each_q, prob_each_p

#functions needed in metrological potential

def get_allqbits_q0p0_T(probs_q0p0,R1,T):
    '''
    probs_q0p0: array with the probability of measuring the bit at g in round R, up to round R1. Shape [q or p][R]
    
    return the probability of all bitstrings with length T, that is only the q bits, for a given displacement q0,p0.
    '''
    
    probs_q0p0_q = probs_q0p0[0][1:]
    probs_allbits_T = []
    for i in range(2**(T)):
        b = int_to_r_bits(i,T)
        prob = probability_bitstring_probs(b,probs_q0p0_q)
        probs_allbits_T.append(prob)
    return np.array(probs_allbits_T)

def get_allpbits_q0p0_T(probs_q0p0,R1,T):
    '''
    probs_q0p0: array with the probability of measuring the bit at g in round R, up to round R1. Shape [q or p][R]
    
    return the probability of all bitstrings with length T, that is only the q bits, for a given displacement q0,p0.
    '''
    
    probs_q0p0_q = probs_q0p0[1][1:]
    probs_allbits_T = []
    for i in range(2**(T)):
        b = int_to_r_bits(i,T)
        prob = probability_bitstring_probs(b,probs_q0p0_q)
        probs_allbits_T.append(prob)
    return np.array(probs_allbits_T)

def probability_bitstring_probs(bitstring, probs):
    """
    Calculates the probability of a given bitstring.
    bitstring (str): The binary bitstring.
    probs (list): The probability of getting a 0 in each bit.
    Returns:
    float: The probability of the bitstring.
    """
    prob = 1
    for i, b in enumerate(bitstring):
        prob *= probs[i] if b == '0' else 1 - probs[i]
    return prob

def get_probabilities_bitsT(probs_q0p0_allbits,R,T):
    '''
    probs_q0p0_allbits: array with the probability of measuring the bit at g in round r, up to round R. Shape [2**(2*T)]. q0, p0 fixed.
    returns: the probability of measuring the bits at g in round r, up to round T. Shape [2**(2*T)]
    '''
    probs_allbits_q0p0 = []
    for bT in range(2**(2*T)):
        prob_bT =0
        for bR in range(2**(2*R)):
            bitT = int_to_r_bits(bT,2*T)
            bitR = int_to_r_bits(bR,2*R)
            if bitT == bitR[:2*T]:
                prob_bT += probs_q0p0_allbits[bR]
        probs_allbits_q0p0.append(prob_bT)
    return np.array(probs_allbits_q0p0)

def get_probabilities_qbitsT(probs_q0p0_qbits,R,T):
    '''
    probs_qp0_qbits: array with the probabilities of all q bitstrings, up to round R1. Shape [q0][2**(R)].
    returns: the probability of measuring the bits at g in round r, up to round T. Shape [2**(T)]
    '''
    probs_Tbits_q0p0 = []
    for bT in range(2**(T)):
        prob_bT =0
        for bR in range(2**(R)):
            bitT = int_to_r_bits(bT,T)
            bitR = int_to_r_bits(bR,R)
            if bitT == bitR[:T]:
                prob_bT += probs_q0p0_qbits[bR]
        probs_Tbits_q0p0.append(prob_bT)
    return np.array(probs_Tbits_q0p0)

def marginalize_probs(probs, T):
    """
    Marginalize a probability array over the last (8-T) bits.

    Args:
        probs (np.ndarray): Array of shape (256,) with probabilities for all 8-bit bitstrings.
        T (int): Number of bits to keep (T < 8).

    Returns:
        np.ndarray: Array of shape (2**T,) with marginalized probabilities.
    """
    assert probs.shape == (256,), "Input probs must have shape (256,)"
    if T==8:
        return probs
    elif T > 8:
        raise ValueError("T must be less than or equal to 8")
    else:
        probs = probs.reshape([2]*8)  # shape (2,2,2,2,2,2,2,2)
        # Sum over the last (8-T) axes
        marginalized = probs.copy()
        for axis in range(T, 8):
            marginalized = marginalized.sum(axis=-1)
        return marginalized.flatten()