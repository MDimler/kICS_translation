# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 22:35:52 2023

@author: Martin
"""
import numpy as np
import scipy
import math

def kICS(J, varargin):
    norm_lag = 0
    use_norm = 1
    use_WKT = 1
    use_time_fluct = 1
    time_win = 0
    force_even = 0
    for ii in range(0,len(varargin),2):
        if varargin[ii] in ('normByLag','normalizeByLag','normalizeByNthLag','tauLagNorm'):
            if varargin[ii+1].isnumeric() and varargin[ii+1] in (0,1):
                norm_lag = varargin[ii+1]
            elif varargin[ii+1] in ("none", "noNorm"):
                use_norm = 0
            else:
                print(f"Unknown option for {varargin[ii]}, using default options.")
        elif varargin[ii] in ('useWKT', 'WKT'):
            if varargin[ii+1].isnumeric() and varargin[ii+1] in (0,1):
                use_WKT = varargin[ii+1]
            else:
                print(f"Unknown option for {varargin[ii]}, using default options.")
        elif varargin[ii] in ('useTimeFluct', 'subTempMean', 'subMean'):
            if varargin[ii+1].isnumeric() and varargin[ii+1] in (0,1):
                use_time_fluct = varargin[ii+1]
            else:
                print(f"Unknown option for {varargin[ii]}, using default options.")
        elif varargin[ii] in ('timeWindow', 'timeWin', 'window'):
            if varargin[ii+1].isnumeric() and varargin[ii+1]>0 and len(varargin[ii+1]<=2):
                time_win = 1
                win_k = varargin[ii+1]
            else:
                print(f"Unknown option for {varargin[ii]}, using default options.")
        elif varargin[ii] in ('even', 'forceEven', 'mirrorMovie'):
            force_even = 1
        else:
            print(f"Unknown varagin input {varargin[ii]}")
        
    size_y = np.size(J,0)
    size_x = np.size(J,1)
    T = np.size(J,2)
    
    if use_time_fluct and not time_win:
        J_fluct = J - np.mean(J, 2)
    elif time_win:
        N = win_k-1
        J_fluct = J-scipy.signal.lfilter(np.ones(N)/N, [1], J, 2)[N:]
    else:
        J_fluct = J
    
    if force_even:
        J_mirror = np.array([[J_fluct, np.fliplr(J_fluct[:,0:-1,:])]
                             ,[np.flipud(J_fluct[:-1,:-1,:]), np.rot90(J_fluct[:-1,:,:], 2)]])
        J_k = np.fft2(J_mirror)
    else:
        J_k = np.fft2(J_fluct)

    if use_WKT:
        if math.floor(math.log2(T)) == math.log2(T):
            T_pad = 2*T
        else:
            T_pad = 2**(math.ceil(math.log(T, 2)))
        
        F_J_k = np.fft(J_k, T_pad, 2)
        r_k = np.ifft((F_J_k)*np.conjugate(F_J_k, [], 2))
        r_k = np.delete(r_k, slice(T,len(r_k)),2)
    else:
        r_k = np.zeros((size_x, size_y, T))
        for tau in range(T):
            r_k[:,:,tau] = np.sum(J_k[:,:,0:T-tau].conj()*J_k[:,:,tau:T], axis = 2)
    for tau in range(T):
        r_k[:,:,tau] = 1/(T-tau)*np.fft.fftshift(r_k[:,:,tau])
    
    if use_norm:
        phi_k = r_k/r_k[:,:,norm_lag]
    else:
        phi_k = r_k
    
    return phi_k