# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 11:36:40 2023

@author: Martin
"""
import numpy as np

def getKSqVector(r_k, varargin="", nargout = False):
    
    ksq_min = 'min'
    ksq_max = 'max'
    use_zero = 0
    for i in range(len(varargin)):
        try:
            if varargin[i].lower() == 'kSqMin'.lower():
                if isinstance(varargin[i+1],(int,float,np.float64)):
                    ksq_min = varargin[i+1]
                elif varargin[i+1].lower() == 'min'.lower():
                    pass # do nothing; min set by default
                else:
                    print(f"Unknown option for {varargin[i]}, using default options.")
            elif varargin[i].lower() == 'kSqMax'.lower():
                if isinstance(varargin[i+1],(int,float,np.float64)):
                    ksq_max = varargin[i+1]
                elif varargin[i+1].lower() == 'max'.lower():
                    pass # do nothing; max set by default
                else:
                    print(f"Unknown option for {varargin[i]}, using default options.")
            elif varargin[i].lower() in ('useZero'.lower(), 'includeZero'.lower()):
                if isinstance((varargin[i+1]),int) and varargin[i+1] in (0,1):
                    use_zero = varargin[i+1]
                else:
                    print(f"Unknown option for {varargin[i]}, using default options.")
        except:
            pass

    # compute spatial dimestions of input array
    size_y = np.size(r_k, 0)
    size_x = np.size(r_k, 1)

    # get 1-d vector of corresponding lattice points
    # treat even & odd cases separately
    if size_x%2 == 0:
        xgv = np.arange(-size_x/2, size_x/2)
    else:
        xgv = np.arange(-(size_x-1)/2, (size_x-1)/2+1)

    if size_y%2 == 0:
        ygv = np.arange(-size_y/2, size_y/2)
    else:
        ygv = np.arange(-(size_y-1)/2, (size_y-1)/2+1)
    # create xy-lattices
    xm, ym = np.meshgrid(xgv,ygv)
    # equivalent integer lattice to k-squared lattice
    int_lattice_sqrd = (xm*size_y)**2 + (ym*size_x)**2
    # norm squared of lattice
    k_lattice_sqrd = (2*np.pi)**2 * ((xm/size_x)**2 + (ym/size_y)**2)
    # unique values occuring in lattice sorted into vector
    ksq = (2*np.pi)**2 / (size_x*size_y)**2 * np.unique(int_lattice_sqrd)
    # lowest index, i, which satisfies kSqVector(i) >= kSqMin
    if isinstance(ksq_min,str) and ksq_min.lower() == 'min'.lower():
        iksq_min = 0
        ksq_min = ksq[0]
    else:
        for j in range(len(ksq)):  # Exchanged for iksq_min = find(ksq >= ksq_min,1,'first')
            if ksq[j] >= ksq_min:
                iksq_min = j
                break
    
    # highest index, j, which satisfies kSqVector(j) <= kSqMax
    if isinstance(ksq_max,str) and ksq_max.lower() == 'max'.lower():
        iksq_max = len(ksq)
        ksq_max = ksq[-1]
    else:
        for j in range(len(ksq)-1,-1,-1):  # Exchanged for iksq_max = find(ksq <= ksq_max,1,'last');
            if ksq[j] <= ksq_max:
                iksq_max = j
                break
    ksq_sub = ksq[iksq_min:iksq_max]
    iksq_sub = np.arange(iksq_min, iksq_max)
    if nargout:
        # get corresponding lattice indices in the range [ksq_min,ksq_max]
        # lattice_inds = np.where(k_lattice_sqrd <= ksq_max and k_lattice_sqrd >= ksq_min)
        lattice_inds = np.where((k_lattice_sqrd <= ksq_max) & (k_lattice_sqrd >= ksq_min))
    if use_zero == 0 and ksq_sub[0] == 0:
        ksq_sub = np.delete(ksq_sub, 0)
        iksq_sub = np.delete(iksq_sub, 0)
        if nargout:
            lattice_inds = np.delete(lattice_inds, 0)
    if nargout:
        return lattice_inds
    return (ksq_sub, iksq_sub)
