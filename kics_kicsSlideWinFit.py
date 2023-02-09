# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 12:46:14 2023

@author: Martin
"""
import numpy as np

from kics_kicsSlideWinFn import kICSSlideWinFn

def kICSSlideWinFit(params, kSq, tauVector, T_s, varargin):
    all_vars = ('D','r','K','frac','Ts','ksq','tau')
    all_vals = (params[0],params[1],params[2],params[3],T_s,kSq,tauVector)
    s = {all_vars[0] : all_vals[0]}
    for i in range(1,len(all_vars)): s[all_vars[i]]=all_vals[i]

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
        # elif varargin[i].lower() in ('symVars'.lower(), 'symsVars'.lower()):
        #     if (isinstance(varargin[i+1], np.ndarray) 
        #         and all(isinstance(j, str) for j in varargin[i+1])):
        #         sym_vars = varargin[i+1]
        #     else:
        #         print(f"Unknown option for {varargin[i]}, using default options.")
    #s = setSymVars(all_vars, all_vals, sym_vars)
    
    tauGrid, kSqGrid = np.meshgrid(tauVector,kSq)
    tauGrid = tauGrid.astype(np.float64)
    # fit function computation
    static_term, diff_term, static_term_norm, diff_term_norm = kICSSlideWinFn(s,kSqGrid,tauGrid)
    # PSF in k-space
    # Ik = exp(-s.w0^2.*kSqGrid/8);
    # Ik0 = exp(-s.w0^2.*kSq(1)/8);
    

    # normalized correltaion function
    F = ((s['frac']*diff_term+(1-s['frac'])*static_term)
        /(s['frac']*diff_term_norm+(1-s['frac'])*static_term_norm))
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
        out = np.double(err) # evtl Falsch double(err)???
        if np.isnan(out):
            print(f"2Function is undefined for params: {params}.")
            
    return out




