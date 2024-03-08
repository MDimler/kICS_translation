# -*- coding: utf-8 -*-
"""This module contains the function timeIntkICSFit.

Created on Mon Jan 23 16:59:55 2023
@author: Martin
"""

import numpy as np

from kics_timeIntkICsFn import timeIntkICsFn

def timeIntkICSFit(params,kSq,tauVector,varargin):
        """This function contains the normalised correlation function that is used for fitting the autocorrelation.

    Inputs:
    params:         tupel containing 4 floats. Parameter used for fitting.
    kSq:            1 dimensional float numpy array containing the |k|^2-vector.
    tauVector:      1 dimensional int array containing the time lags in frames to be fitted.
    varagin:        Tupel containing additional options and maybe the autocorrelation.
    Outputs:        
    if errBool==True:
    err:            float corresponding to error of the correlation fuction to the autocorrelation.
    else:
    out:            The value of the normalised correlation function.        
    """
    errBool = 0
    for i in range(len(varargin)):
        try:
            if varargin[i].lower() in ('error'.lower(), 'err'.lower(),
                                       'residual'.lower(), 'res'.lower()):
                errBool = 1
                ydata = varargin[i+1]
        except:
            pass

    # Grouping of the parameters
    s = {'diffusion':params[0],'r':params[1],'K':params[2],'frac':params[3]}
    tauGrid, kSqGrid = np.meshgrid(tauVector+1,kSq)
    #print(np.shape(tauVector), np.shape(kSq), np.shape(tauGrid), np.shape(kSqGrid), tauGrid[-1,-1], kSqGrid[-1,-1])
    tauGrid = tauGrid.astype(np.float64)
    kSqGrid = kSqGrid.astype(np.float64)
    # fit function computation
    diff_term,static_term,diff_term_norm,static_term_norm = timeIntkICsFn(s,tauGrid,kSqGrid)
    # normalization
    F_norm = s['frac']*diff_term_norm+(1-s['frac'])*static_term_norm
    # normalized correlation function

    F = (s['frac']*diff_term+(1-s['frac'])*static_term)/F_norm

    # Calculation of the norm point-by-point.
    if errBool:
        err = np.sqrt(np.sum((F-ydata)**2))
    
    if not errBool:
        out = F
        if np.isnan(out.any()):
            print(f"1Function is undefined for params: {params}.")
    else:
        out = np.double(err)
        if np.isnan(out):
            print(f"2Function is undefined for params: {params}.")
    
    return out

