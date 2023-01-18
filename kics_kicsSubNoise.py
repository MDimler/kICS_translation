# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:46:52 2023

@author: Martin
"""

# Outputs noise subtracted extended kICS correlation at time lag 0. ksq_min
# and ksq_max are large k-squared values, where the autocorrelation at
# tau=0 has sufficiently decayed and can be considered as the contribution
# from the noise. All k-squared values between [ksq_min,ksq_max] are
# averaged and subtracted from the zero time-lag.

# INPUT PARAMETERS

# r_k:          output from kICS(...)
# ksq_min:      minimum k-squared bound for averaging
# ksq_max:      maximum k-squared bound for averaging

# OUTPUT PARAMTERS

# r_k_0_sub:    noise-subtracted extended kICS autocorrelation at tau=0

import numpy as np

from kics_getKSqVector import getKSqVector

def kICSSubNoise(r_k,ksq_min,ksq_max):
    # get k-squared values between [ksq_min,ksq_max] for averaging
    (_,_,noise_inds) = getKSqVector(r_k, ('kSqMin', ksq_min, 'kSqMax', ksq_max), True)
    r_k_0 = r_k[:,:,0]
    
    # subtract averaged noise value estimate from autocorrelation at tau=0
    r_k_0_sub = r_k[:,:,0] - np.mean(r_k_0[noise_inds])
    return r_k_0_sub
