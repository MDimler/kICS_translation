# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 10:55:45 2023

@author: Martin
"""
import numpy as np

def circular(r_k_norm, varargin):
    
    # compute size of input array over every dimension
    size_y = len(r_k_norm[0,:])
    size_x = len(r_k_norm[1,:])
    T = len(r_k_norm[2,:])
    
    # 1-d vector of corresponding lattice points
    # treat even and odd cases separately
    
    if size_x%2 == 0:
        xgv = np.arrange(-size_x//2, size_x//2-1)
    else:
        xgv = np.arrange(-(size_x-1)/2, (size_x-1)/2)

    if size_y%2 == 0:
        ygv = np.arrange(-size_y//2, size_x//2-1)
    else:
        ygv = np.arrange(-(size_y-1)//2, (size_y-1)//2)
     
    # create xy-lattices with zeros at the center position
    xm, ym = np.meshgrid(xgv, ygv)
    # norm squared of integer lattice equivalent to ksq lattice
    int_lattice_sqrd = (xm*size_y)**2 + (ym*size_x)**2
    # unique values occuring in integer lattice sorted
    lattice_nums = np.unique(int_lattice_sqrd)
    
    # number of values in "lattice_nums"      
    l_nums = len(lattice_nums)
    
    r_k_circ = np.zeros(l_nums,T) # array to fill with circular average of r_k_norm
    unique_inds = np.zeros(l_nums,1)
    for i in range(len(lattice_nums)):
        n = lattice_nums[i]
        inds = np.where(int_lattice_sqrd == n) # indices in "int_lattice_sqrd" with value "n"
        subs_y, subs_x = np.unravel_index(inds, np.size(int_lattice_sqrd()), "F")
        unique_inds[i]=inds[0]
        l_inds = len(inds)
        
        for j in range(l_inds):
            y = subs_y[j]
            x = subs_x[j]
            r_xy = np.reshape(r_k_norm[y,x,:], (1,T))
            r_k_circ[i,:] = r_k_circ[i,:] + r_xy/l_inds
            
    ksq = (2*np.pi)**2 / (size_x*size_y)**2 * lattice_nums
    return r_k_circ, ksq
