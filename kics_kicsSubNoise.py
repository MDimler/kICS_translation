# -*- coding: utf-8 -*-
"""This module contains the function kICSSubNoise.

Created on Thu Jan 12 14:46:52 2023
@author: Martin
"""

import numpy as np

from kics_getKSqVector import getKSqVector

def kICSSubNoise(r_k,ksq_min,ksq_max):
    """ This function gives the subtracted extended kICS correltaion at time lag 0.
    
    Inputs:
    r_k:              2 or 3 dimensional array corresponding to the output from kICS
    ksq_min:          Float corresponding to the minimum |k|^2 bound for averaging
    ksq_max:          Float corresponding to the maximum |k|^2 bound for averaging
    Outputs:
    r_k_0_sub    :    noise-subtracted extended kICS autocorrelation at tau=0
    """
    # get k-squared values between [ksq_min,ksq_max] for averaging
    noise_inds = getKSqVector(r_k, ('kSqMin', ksq_min, 'kSqMax', ksq_max), True)
    r_k_0 = r_k[:,:,0]
    
    # subtract averaged noise value estimate from autocorrelation at tau=0
    r_k_0_sub = r_k[:,:,0] - np.mean(r_k_0[noise_inds])
    
    return r_k_0_sub
