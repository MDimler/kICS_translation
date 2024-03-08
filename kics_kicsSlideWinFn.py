# -*- coding: utf-8 -*-
"""This module contains the function kICSSlideWinFn. 

Created on Mon Jan 16 11:28:13 2023
@author: Martin
"""

import numpy as np
import math

def kICSSlideWinFn(s, kSqGrid, tauGrid):
    """This function does some parts of the calculation of the normalised correlation function.

    Inputs:
    s:            Dictionary containing parameters needed for fitting. I.e. diffusion coefficient parameter, etc.
    kSqGrid, tauGrid: Complex numpy arrays corresponding to the |k|^2-vector and the time lags.
    Outputs:
    tupel         Tupel corresponding to different parts of the normalised correlation function.
    """
    A = s['D'] * kSqGrid

    # static autocorrelation term
    static_term_t = np.exp((-1)*s['K']*((-1)+s['Ts']+ (-1)*tauGrid))
    
    
    static_term = (s['K']**(-2)*((-1)+s['r'])*s['r']*s['Ts']**(-2)*((-1)*np.exp((-1)*s['K']*((-1)+s['Ts']+ \
        (-1)*tauGrid))*(1+np.exp((-2)*s['K']*tauGrid))*(np.exp((-1)*s['K'])+(1+(-1)* \
        np.exp((-1)*s['K']))*s['Ts'])+2*(s['K']*tauGrid+np.exp((-1)*s['K']*tauGrid)*(1+((-1)+ \
        s['Ts'])*s['Ts']+(-1)*((-1)+s['Ts'])*s['Ts']*np.cosh(s['K'])))))
                                                                                                      
    diff_term_part1 =((-1) * np.exp((-1) *A *((-1)+tauGrid)) *(A **2 * np.exp((-1) *s['K'] *((  
               -1)+tauGrid)) *(1+(-2) * np.exp((-1) *A+(-1) *s['K'])+ np.exp((-2) *(A+s['K'])))  
                *((-1)+s['r'])+(-1) *(1+ np.exp((-2) *A)+(-2) * np.exp((-1) *A)) *(  
               A+s['K']) **2 *s['r']) *s['Ts']+ np.exp((-1) *A *tauGrid) *((A+s['K']) **2 *s['r'] *((-1)+ np.exp(
               (-1) *A *s['Ts'])+s['Ts']+(-1) * np.exp((-1) *A) *s['Ts'])+A **2 * np.exp((  
               -1) *s['K'] *tauGrid) *((-1)+s['r']) *(1+(-1) * np.exp(((-1) *A+(-1) *s['K']) *s['Ts'])+(  
               -1) *s['Ts']+ np.exp((-1) *A+(-1) *s['K']) *s['Ts']))+2 *A *(A+s['K']) *(A+A *(1+(-1)  
                * np.exp((-1) *A+(-1) *s['K'])) *(A+s['K']) **(-1) *((-1)+s['r'])+((-1)+ np.exp((-1) *A))
               *s['r']+A **(-1) *((-1)+A+ np.exp((-1) *A)) *s['K'] *s['r']) *(s['Ts']+(  
               -1) *tauGrid)+2 *A *s['K'] *s['r'] *((-1)+ np.exp((-1) *A *(s['Ts']+(-1) *tauGrid))+s['Ts']+(-1)  
                * np.exp((-1) *A) *(s['Ts']+(-1) *tauGrid)+(-1) *tauGrid)+s['K'] **2 *s['r'] *((-1)+ np.exp(
               (-1) *A *(s['Ts']+(-1) *tauGrid))+s['Ts']+(-1) * np.exp((-1) *A) *(s['Ts']+(-1) *tauGrid)  
               +(-1) *tauGrid)+A **2 *((-1)+(-1) * np.exp(((-1) *A+(-1) *s['K']) *(s['Ts']+(-1) *  
               tauGrid)) *((-1)+s['r'])+ np.exp((-1) *A *(s['Ts']+(-1) *tauGrid)) *s['r']+s['Ts']+ np.exp((-1)  
                *A+(-1) *s['K']) *((-1)+s['r']) *(s['Ts']+(-1) *tauGrid)+(-1) * np.exp((-1) *A) *s['r'] *(  
               s['Ts']+(-1) *tauGrid)+(-1) *tauGrid)+(-1) *(A+s['K']) **2 *s['r'] *((-1)+(-1) * np.exp((-1)  
                *A *tauGrid) *((-1)+s['Ts'])+(-1) *s['Ts']+ np.exp((-1) *A *((-1)+tauGrid)) *s['Ts']+ np.exp(
               (-1) *A) *(s['Ts']+(-1) *tauGrid)+tauGrid)+A **2 *((-1)+s['r']) *((-1)+(-1) * np.exp(  
               ((-1) *A+(-1) *s['K']) *tauGrid) *((-1)+s['Ts'])+(-1) *s['Ts']+ np.exp(((-1) *A+(-1) *  
               s['K']) *((-1)+tauGrid)) *s['Ts']+ np.exp((-1) *A+(-1) *s['K']) *(s['Ts']+(-1) *tauGrid)+tauGrid))
    
    diff_term_part2 = ((-1) * np.exp(((-1) *A+(-1) *s['K']) *((-1)+s['Ts']+(-1) *tauGrid)) *((  
                -1)+s['r'])+ np.exp(((-1) *A+(-1) *s['K']) *(s['Ts']+(-1) *tauGrid)) *((-1)+s['r'])+(-1) *  
                 np.exp(((-1) *A+(-1) *s['K']) *tauGrid) *((-1)+s['r'])+ np.exp(((-1) *A+(-1) *  
                s['K']) *(1+tauGrid)) *((-1)+s['r'])+ np.exp((-1) *A *((-1)+s['Ts']+(-1) *tauGrid)) *s['r']+(-1)  
                 * np.exp((-1) *A *(s['Ts']+(-1) *tauGrid)) *s['r']+ np.exp((-1) *A *tauGrid) *s['r']+(-1)  
                 * np.exp((-1) *A *(1+tauGrid)) *s['r']+(-2) *s['K'] *(1+s['r']))
                                                                                       
    diff_term = (A ** (-2) * (A+s['K']) ** (-2) * s['r'] *(np.exp((-1) * A * ((-1)+tauGrid)) * (1+np.exp(
                (-2) *A)+(-2) * np.exp((-1) *A)) *s['K'] *(2 *A+s['K']) *s['r']+(-1) *A **2 *  
                 np.exp(((-1) *A+(-1) *s['K']) *((-1)+tauGrid)) *((1+(-1) * np.exp((-1) *A+  
                (-1) *s['K'])) **2 *((-1)+s['r'])+(-1) * np.exp((-1) *s['K'] *(1+(-1) *tauGrid)) *(1+(  
                -1) * np.exp((-1) *A)) **2 *s['r'])+(-1) *( np.exp((-1) *A *((-1)+tauGrid))  
                 *(1+ np.exp((-2) *A)) *s['K'] *(2 *A+s['K']) *s['r']+(-1) * np.exp((-1) *A *((  
                -2)+tauGrid)+(-1) *s['K'] *((-1)+tauGrid)) *(A **2 * np.exp((-1) *A) *(1+(-1) * np.exp((-1) 
                *A+(-1) *s['K'])) *(1+(-1) * np.exp(((-1) *A+(-1) *s['K']) *s['Ts'])) 
                 *((-1)+s['r'])+ np.exp((-1) *A+(-1) *s['K'] *(1+(-1) *tauGrid)) *((-1) *A **2+  
                 np.exp((-1) *A) *(A+s['K']) **2+ np.exp((-1) *A *s['Ts']) *(1+(-1) * np.exp(
                (-1) *A)) *(A+s['K']) **2+ np.exp((-2) *A) *s['K'] *(2 *A+s['K'])) *s['r'])) *s['Ts'] **(  
                -1)+((-2) *A **3+ np.exp((-1) *A *tauGrid) *(1+(-1) * np.exp((-1) *A))  
                 *(1+ np.exp((-1) *A *((-1)+s['Ts']+(-2) *tauGrid))) *s['K'] **2 *s['r']+(-2) *A *s['K'] *((  
                -1) * np.exp((-1) *A *((-1)+s['Ts']+(-1) *tauGrid))+ np.exp((-1) *A *(s['Ts']+(  
                -1) *tauGrid))+(-1) * np.exp((-1) *A *tauGrid)+ np.exp((-1) *A *(1+tauGrid))+s['K']) *  
                s['r']+A **2 *diff_term_part2) *s['Ts'] **(-1)+s['Ts'] **(-2)  
                 *diff_term_part1))
                                                                                        
                                                                                        
    diff_term_norm = A**(-2)*np.exp(A+s['K'])*(A+s['K'])**(-2)*s['r']*s['Ts']**(-2)*(2*A**3*np.exp( \
        (-1)*A+(-1)*s['K'])*((-1)+s['Ts'])*s['Ts']+(1/2)*A**2*(1+(-1)*np.exp( \
        (-2)*(A+s['K'])))*s['r']*s['Ts']**2+(1/2)*A**2*(1+np.exp((-2)*(A+s['K'])))* \
        s['Ts']*((-4)+4*s['r']+2*s['Ts']+(-3)*s['r']*s['Ts'])+2*A*np.exp((-1)*A+(-1)*s['K']) \
        *s['K']*s['r']*((-2)+np.exp((-1)*A*s['Ts'])+(np.exp((-1)*A))**s['Ts']+2* \
        np.exp((-1)*A*((-1)+s['Ts']))*(1+(-1)*np.exp((-1)*A))*s['Ts']+2* \
        np.exp((-1)*A)*((-1)+s['Ts'])*s['Ts']+((-2)+s['K'])*((-1)+s['Ts'])*s['Ts'])+np.exp(\
        (-1)*A+(-1)*s['K'])*s['K']**2*s['r']*((-2)+(np.exp((-1)*A))**s['Ts']+2*s['Ts']+ \
        (-2)*s['Ts']*(np.exp((-1)*A)+(1+(-1)*np.exp((-1)*A))*s['Ts'])+np.exp(\
        (-1)*A*((-1)+s['Ts']))*(np.exp((-1)*A)+2*(1+(-1)*np.exp( \
        (-1)*A))*s['Ts']))+A**2*((-2)*np.exp((-1)*A+(-1)*s['K'])+np.exp((( \
        -1)*A+(-1)*s['K'])*(1+s['Ts']))+(-2)*np.exp(((-1)*A+(-1)*s['K'])*s['Ts'])*(( \
        -1)+s['r'])*s['Ts']+2*np.exp((-1)*s['K']+(-1)*A*s['Ts'])*s['r']*s['Ts']+((-1)+s['r'])*((-2) \
        +s['Ts'])*s['Ts']+2*np.exp((-1)*A+(-1)*s['K'])*((-1)+s['K'])*((-1)+s['Ts'])*s['Ts']+2* \
        np.exp((-2)*A+(-1)*s['K'])*s['r']*((-1)+s['Ts'])*s['Ts']+np.exp((-2)*(A+s['K'])) \
        *s['Ts']**2+np.exp(((-1)*A+(-1)*s['K'])*(1+s['Ts']))*((-1)+s['r'])*((-1)+2* \
        s['Ts'])+np.exp((-1)*A+(-1)*s['K']+(-1)*A*s['Ts'])*(s['r']+(-2)*s['r']*s['Ts'])+np.exp(\
        (-1)*A+(-1)*s['K'])*s['r']*(np.exp((-1)*A*s['Ts'])+(-1)*np.exp((( \
        -1)*A+(-1)*s['K'])*s['Ts'])+2*s['K']*((-1)+s['Ts'])*s['Ts'])))
    
    static_term_norm = 2*s['K']**(-2)*((-1)+s['r'])*s['r']*s['Ts']**(-2)*(1+np.exp((-1)*s['K']*((-1)+s['Ts'])) \
        *(np.exp((-1)*s['K'])*((-1)+s['Ts'])+(-1)*s['Ts'])+s['Ts']*((-1)+s['K']+s['Ts']+(-1)*s['K']* \
        s['Ts'])+np.exp((-1)*s['K'])*(s['Ts']+(-1)*s['Ts']**2))
    
    return (static_term,diff_term,static_term_norm,diff_term_norm)

