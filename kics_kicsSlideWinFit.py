# -*- coding: utf-8 -*-
"""This module contains the function kICSSlideWinFit.

Created on Mon Jan 16 12:46:14 2023
@author: Martin
"""
import numpy as np

from kics_kicsSlideWinFn import kICSSlideWinFn

def kICSSlideWinFit(params, kSq, tauVector, T_s, varargin):
    """This function contains the normalised correlation function with consideration to the time window that is used for fitting the autocorrelation.

    Inputs:
    params:         tupel containing 4 floats. Parameter used for fitting.
    kSq:            1 dimensional float numpy array containing the |k|^2-vector.
    tauVector:      1 dimensional int array containing the time lags in frames to be fitted.
    T_S:            int corresponding to the time window.
    varagin:        Tupel containing additional options and maybe the autocorrelation.
    Outputs:        
    if errBool==True:
    err:            float corresponding to error of the correlation fuction to the autocorrelation.
    else:
    out:            The value of the normalised correlation function.        
    """

    # Grouping of parameters
    all_vars = ('D','r','K','frac','Ts','ksq','tau')
    all_vals = (params[0],params[1],params[2],params[3],T_s,kSq,tauVector)
    s = {all_vars[0] : all_vals[0]}
    for i in range(1,len(all_vars)): s[all_vars[i]]=all_vals[i]

    # Readout of the autocorrelation from varargin
    errBool = 0
    #sym_vars = np.array([''])
    for i in range(len(varargin)):
        try:
            if varargin[i].lower() in ('error'.lower(), 'err'.lower(),
                                       'residual'.lower(), 'res'.lower()):
                errBool = 1
                ydata = varargin[i+1]
        except:
            pass
    
    
    tauGrid, kSqGrid = np.meshgrid(tauVector+1,kSq)
    tauGrid = tauGrid.astype(np.float64)
    kSqGrid = kSqGrid.astype(np.float64)
    # fit function computation
    static_term, diff_term, static_term_norm, diff_term_norm = kICSSlideWinFn(s,kSqGrid,tauGrid)
   
    # Calculation of the normalized correltaion function
    F = ((s['frac']*diff_term+(1-s['frac'])*static_term)
        /(s['frac']*diff_term_norm+(1-s['frac'])*static_term_norm))
    
    # Calculation of the norm point-by-point.
    if errBool:
        err = np.sqrt(np.sum((F-ydata)**2))

    if not errBool:
        out = F
        if np.isnan(out.all()):
            print(f"1Function is undefined for params: {params}.")
    else:
        out = np.double(err)
        if np.isnan(out):
            print(f"2Function is undefined for params: {params}.")    
    return out
