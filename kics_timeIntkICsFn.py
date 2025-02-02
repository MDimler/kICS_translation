# -*- coding: utf-8 -*-
"""This module contains the function timeIntkICsFn.

Created on Mon Jan 23 17:12:49 2023
@author: Martin
"""

import numpy as np

def timeIntkICsFn(s,tauGrid,kSqGrid):
    """This function does some parts of the calculation of the normalised correlation function.

    Inputs:
    s:            Dictionary containing parameters needed for fitting. I.e. diffusion coefficient parameter, etc.
    kSqGrid, tauGrid: Complex numpy arrays corresponding to the |k|^2-vector and the time lags.
    Outputs:
    tupel         Tupel corresponding to different parts of the normalised correlation function.
    """
    A = s['diffusion'] * kSqGrid
    # diffusing autocorrelation factor
    diff_term = (np.exp(-(s['K']+A)*(1+tauGrid))*(np.exp(s['K']*(1+tauGrid))
                *(np.exp(A)-1)**2*s['r']/A**2+(1-s['r'])/(A+s['K'])**2
                *(np.exp(A+s['K'])-1)**2))
    # static autocorrelation term
    static_term = ((1-s['r'])/(s['K'])**2*np.exp(-s['K']*(1+tauGrid))
                   *(np.exp(s['K'])-1)**2)
    # diffusing autocorrelation term (norm)
    diff_term_norm = (2*np.exp(-(s['K']+A))*1/(A**2*(A+s['K'])**2)
                    *(A**2*(1-s['r'])+np.exp(s['K'])*((A+s['K'])**2
                    *s['r']+np.exp(A)*(A**2*(A+s['K']-1)+s['K']
                    *s['r']*(-s['K']+A*(A+s['K']-2))))))
    # static autocorrelation term (norm)
    static_term_norm = (2*np.exp(-s['K'])*(1-np.exp(s['K'])*(1-s['K']))
                        *(1-s['r'])/(s['K'])**2)
    # print(np.shape(diff_term_norm), np.shape(kSqGrid))
    return (diff_term,static_term,diff_term_norm,static_term_norm)

