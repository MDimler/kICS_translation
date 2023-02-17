# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 16:59:55 2023

@author: Martin
"""
import numpy as np
from kics_timeIntkICsFn import timeIntkICsFn

def timeIntkICSFit(params,kSq,tauVector,varargin):
    errBool = 0
    for i in range(len(varargin)):
        # return output error (input calculated ACF)
        try:
            if varargin[i].lower() in ('error'.lower(), 'err'.lower(),
                                       'residual'.lower(), 'res'.lower()):
                errBool = 1
                ydata = varargin[i+1]
        except:
            pass

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
    
    # the best measure for the error, so far, seems to be to calculate the LS
    # of each curve in tau individually, and then sum it. This is instead of
    # calculating the norm point-by-point.
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

