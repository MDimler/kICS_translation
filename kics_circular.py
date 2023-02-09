# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 10:55:45 2023

@author: Martin
"""
import numpy as np

def circular(r_k_norm, varargin=""):
    
    # compute size of input array over every dimension
    size_y = np.size(r_k_norm,0)
    size_x = np.size(r_k_norm,1)
    try:
        T = np.size(r_k_norm,2)
    except:
        T = 1
    # 1-d vector of corresponding lattice points
    # treat even and odd cases separately
    
    if size_x%2 == 0:
        xgv = np.arange(-size_x//2, size_x//2-1)
    else:
        xgv = np.arange(-(size_x-1)/2, (size_x-1)/2)

    if size_y%2 == 0:
        ygv = np.arange(-size_y//2, size_x//2-1)
    else:
        ygv = np.arange(-(size_y-1)//2, (size_y-1)//2)
     
    # create xy-lattices with zeros at the center position
    xm, ym = np.meshgrid(xgv, ygv)
    # norm squared of integer lattice equivalent to ksq lattice
    int_lattice_sqrd = (xm*size_y)**2 + (ym*size_x)**2
    # unique values occuring in integer lattice sorted
    lattice_nums = np.unique(int_lattice_sqrd)
    
    # number of values in "lattice_nums"      
    l_nums = len(lattice_nums)
    
    r_k_circ = np.zeros((l_nums,T),dtype=complex) # array to fill with circular average of r_k_norm
    # unique_inds = np.zeros((l_nums,1))
    for i in range(len(lattice_nums)):
        n = lattice_nums[i]
        inds = np.where(int_lattice_sqrd == n) # indices in "int_lattice_sqrd" with value "n"
        # subs_xy = np.unravel_index(inds, np.size(int_lattice_sqrd), "F")
        # print(inds)
        # print(subs_xy)
        # subs_x = subs_xy[0][0]
        # subs_y = subs_xy[0][1]
        # print(subs_x, subs_y)
        # unique_inds[i]=inds[0]
        l_inds = len(inds[0])
        for j in range(l_inds):
            y = inds[0][j]
            x = inds[1][j]

            try:
                r_xy = np.reshape(r_k_norm[x,y,:], (1,T)) # Or r_k_norm[y,x,:]
            except:
                r_xy = np.reshape(r_k_norm[x,y], (1,T)) # Or r_k_norm[y,x,:]
            r_k_circ[i,:] = r_k_circ[i,:] + r_xy/l_inds
            
            
    ksq = (2*np.pi)**2 / (size_x*size_y)**2 * lattice_nums
    return r_k_circ, ksq
