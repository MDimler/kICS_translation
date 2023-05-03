# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 22:35:52 2023

@author: Martin
"""
import numpy as np
import scipy
import math

def kICS(J, varargin):
    
    #  time lag to normalize by
    norm_lag = 0
    #  logical for normalization
    use_norm = 1
    #  use Wiener-Khinchin theorem; note this technically should
    #  not be used for non-stationary processes in time (e.g.
    #  photobleaching)
    use_WKT = 1
    #  determines whether to subtract by the temporal mean of the image series.
    #  Note that subtracting by the spatial mean would leave the spatial Fourier
    #  transform unchanged
    use_time_fluct = 1
    #  determines whether to subtract by the local temporal mean of the image
    #  series using a temporal window that starts at each respective time point
    time_win = 0
    #  extend images periodically in an even manner. This is recommended when
    #  the data is not intrinsically periodic across its boundaries (e.g. real
    #  data). Note using discrete cosine transform (DCT) is equivalent to this
    #  method, but is not used here to maintain convention and to enable user to
    #  perform FFT on result
    force_even = 0
    
    for ii in range(0,len(varargin),2):
        if varargin[ii] in ('normByLag','normalizeByLag','normalizeByNthLag','tauLagNorm'):
            if isinstance((varargin[ii+1]),int) and varargin[ii+1] in (0,1):
                norm_lag = varargin[ii+1]
            elif varargin[ii+1] in ("none", "noNorm"):
                use_norm = 0
            else:
                print(f"Unknown option for {varargin[ii]}, using default options.")
        elif varargin[ii] in ('useWKT', 'WKT'):
            if isinstance((varargin[ii+1]),int) and varargin[ii+1] in (0,1):
                use_WKT = varargin[ii+1]
            else:
                print(f"Unknown option for {varargin[ii]}, using default options.")
        elif varargin[ii] in ('useTimeFluct', 'subTempMean', 'subMean'):
            if isinstance((varargin[ii+1]),int) and varargin[ii+1] in (0,1):
                use_time_fluct = varargin[ii+1]
            else:
                print(f"Unknown option for {varargin[ii]}, using default options.")
        elif varargin[ii] in ('timeWindow', 'timeWin', 'window'):
            if isinstance(varargin[ii+1],(int,float,np.ndarray,np.float64)) and varargin[ii+1]>0 and np.shape(varargin[ii+1])<=(2,2):
                # set windowing boolean to 1
                time_win = 1
                # window size specification
                win_k = varargin[ii+1]
            else:
                print(f"Unknown option for {varargin[ii]}, using default options.")
        elif varargin[ii] in ('even', 'forceEven', 'mirrorMovie'):
            force_even = 1
        else:
            print(f"Unknown varagin input {varargin[ii]}")
    

    J = np.double(J)
    
    size_y = np.size(J,0)
    size_x = np.size(J,1)
    T = np.size(J,2)

    if use_time_fluct and not time_win:
        # subtract by temporal mean
        J_mean = np.tile(np.mean(J, 2),(np.size(J,2),1,1))
        J_mean = np.moveaxis(J_mean,0,-1)
        J_fluct = J - J_mean
    elif time_win:
        # subtract by local temporal mean
        N = win_k-1
        #J_fluct = J-scipy.signal.lfilter(np.ones(N)/N, [1], J, 2)
        # J_fluct = J-scipy.signal.lfilter(np.ones(N)/N, [1], J, 2)
        #J_fluct = J-scipy.ndimage.uniform_filter1d(J,size=N,axis=2,mode='nearest')
        J_fluct = J-scipy.ndimage.uniform_filter1d(J,size=N,axis=2,
                                                   mode='nearest', origin=(-win_k+2)//2)
    else:
        # no fluctuations
        J_fluct = J
    
    if force_even:
        J_mirror = np.array([[J_fluct, np.fliplr(J_fluct[:,0:-2,:])]
                             ,[np.flipud(J_fluct[:-2,:-2,:]), np.rot90(J_fluct[:-2,:,:], 2)]])
        J_k = np.fft.fft2(J_mirror,axes=(0,1))
    else:
        J_k = np.fft.fft2(J_fluct,axes=(0,1))


    if use_WKT:
        if math.floor(math.log2(T)) == math.log2(T):
            T_pad = 2*T
        else:
            T_pad = 2**(math.ceil(math.log2(T)))
        
        F_J_k = np.fft.fft(J_k, T_pad, 2)
        r_k = np.fft.ifft((F_J_k)*np.conjugate(F_J_k),axis=2)
        r_k = r_k[:,:,:T]
    else:
        r_k = np.zeros((size_x, size_y, T))
        for tau in range(T):
            r_k[:,:,tau] = np.dot(J_k[:,:,0:T-tau],J_k[:,:,tau:T], axis = 2)

    for tau in range(T):
        r_k[:,:,tau] = 1/(T-tau)*np.fft.fftshift(r_k[:,:,tau])

    if use_norm:
        phi_k = r_k/r_k[:,:,norm_lag]
    else:
        phi_k = r_k
    return phi_k