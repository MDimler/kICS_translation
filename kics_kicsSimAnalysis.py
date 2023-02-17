# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:44:41 2023

@author: Martin
"""

# this code is for plotting and fitting a TICS blink/bleach autocorrelation
# including plots and fits for blink/bleach regimes

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pickle # speichern der Variablen
import cv2
import sys

from kics_kics import kICS
from kics_kicsSubNoise import kICSSubNoise
from kics_circular import circular
from kics_getKSqVector import getKSqVector
from kics_kicsSlideWinFit import kICSSlideWinFit
from kics_timeIntkICsFit import timeIntkICSFit
#%%
# input variables

# time window
use_time_win = 1
time_win = 200

# logical for including windowing in fit model
win_fit = 1

# tau range to fit
tauVector = np.arange(10)
# tau range to plot
plotTauLags = np.arange(5)

# min/max bounds on |k|^2
kSqMin = np.spacing(1)
kSqMax = 1.8

# min/max bounds on |k|^2
ksq_min_noise = 10
ksq_max_noise = 15

# number of fits to compare
n_fits = 5

# lower/upper bounds on fit parameters, [D,rho_on,K,p_D]
# lb = np.spacing(1)*np.ones((1,4))
# ub = [np.inf, 1, np.inf, 1]
bounds = [(np.spacing(1),np.inf), # np.inf
          (np.spacing(1),1),
          (np.spacing(1),np.inf), # np.inf
          (np.spacing(1),1),
          ]


#  rois to consider (one per row; form [x0,y0,width,height])

roi = np.array([[0,0,63,63],
        [64,64,63,63],
        [0,64,63,63],
        [64,0,63,63]
        ])

# import of images
test1 = False
if test1:
    filename = 'D_5_kon_0.1_koff_0.9_frac_diff_0.7_kp_0.0001.tif'
else:
    filename = 'D_0.01_kon_1_koff_0.7_frac_diff_0.65_kp_0.0001.tif.tif'

im = cv2.imreadmulti(filename,[],cv2.IMREAD_UNCHANGED)
J = np.asarray(im[1],dtype = np.float64)
J = np.moveaxis(J,0,-1)

T = np.size(J,2)
nPtsFitPlot = int(1e3)

# main code

# Ordner???

for i in range(np.size(roi,0)):
    # roi image series
    J_roi = np.copy(J[roi[i, 1]:roi[i, 1]+roi[0, 3]+1,
                    roi[i, 0]:roi[i, 0]+roi[i, 2]+1, :])

    # compute kICS autocorr
    # tic
    
    #kICS autocorrelation function (ACF)
    if use_time_win:
        time_win_varargin = ('timeWin', time_win)
    else:
        time_win_varargin = ()
    
    r_k = kICS(J_roi, ('normByLag', 'none', *time_win_varargin))
    
    r_k_0_sub = kICSSubNoise(np.copy(r_k), ksq_min_noise, ksq_max_noise)
    
    # Circular Averaging
    r_k_0_circ,_ = circular(np.copy(r_k_0_sub[:,:]))
    r_k_circ,_ = circular(r_k)
    # get and cut |k|^2 vector
    kSqVector, kSqInd = getKSqVector(J_roi)
    kSqVectorSubset, kSqSubsetInd = getKSqVector(
        J_roi, ('kSqMin', kSqMin, 'kSqMax', kSqMax))

    # cut autocorrelation
    
    r_k_circ_cut = np.array([r_k_circ[i][tauVector+1] for i in kSqSubsetInd])
    # cut normalization
    r_k_0_circ_cut = r_k_0_circ[kSqSubsetInd,0]
    
    # normalization
    r_k_norm = np.zeros(np.shape(r_k_circ_cut))
    for i_c in range(len(r_k_0_circ_cut)):
        for j_c in range(len(r_k_circ_cut[0])):
            r_k_norm[i_c, j_c] = np.real(
                r_k_circ_cut[i_c, j_c]/r_k_0_circ_cut[i_c])

    # Autocorrelation and fit plot
    # fit function for entire autocorrelation
    if win_fit:
        k_p = np.spacing(1)
        k_p = 200
        err = lambda params: kICSSlideWinFit(params,kSqVectorSubset,tauVector,k_p,(T,time_win+1,'err',r_k_norm,'symvars',np.array([''])))
        fit_fun = lambda params, ksq, tau: kICSSlideWinFit(params,ksq,tau,k_p,(T,time_win+1,'symvars',np.array([''])))
    else:
        err = lambda params: timeIntkICSFit(params,kSqVectorSubset,tauVector,('err',r_k_norm,'symvars',np.array([''])))
        fit_fun = lambda params, ksq, tau: timeIntkICSFit(params,ksq,tau,('symvars',np.array([''])))
    
    params_guess = np.random.rand(4)
    opt_p = optimize.minimize(err,params_guess,bounds=bounds)
    print(opt_p.get('message'), opt_p.get('success'))
    opt_params = opt_p.get('x')
    err_min = opt_p.get('fun')
    # fit n_fits times
    i_fit = 1
    while i_fit <= n_fits-1:
        try:
            params_guess = np.random.rand(4)
            opt_p = optimize.minimize(err,params_guess,bounds=bounds)
            if err_min > opt_p.get('fun'):
                err_min = opt_p.get('fun')
                opt_params = opt_p.get('x')
            i_fit += 1
        except:
            pass
    #
    # plot simulation data
    #
    plot_idx = plotTauLags  # Changed from [~,plot_idx] = ismember(plotTauLags,tauVector); becomes important if plotTauLags or tauVector changes
    
    # |k|^2 for plotting best fit/theory curves
    ksq2plot = np.linspace(kSqVectorSubset[0],kSqVectorSubset[-1],nPtsFitPlot)
    
    fig, ax = plt.subplots()
    
    h_sim_data = np.zeros((1,len(plotTauLags)))
    
    for tauInd in range(len(plotTauLags)):
        ax.scatter(kSqVectorSubset,r_k_norm[:,plot_idx[tauInd]], label=r'$\tau = \ %d $' %(plotTauLags[tauInd]), s=10)

    ax.set_xlabel(r'$|k|^2$ (pixels$^{-2}$)')
    ax.set_ylabel(r'$\tilde{\Phi}(|$k$|^2,\tau)$')
    ax.legend()
    ax.set_xlim([kSqVectorSubset[0], kSqVectorSubset[-1]])
    # ax.set_ylim()
    
    # # test
    # opt_params = np.array([0.01, 0.588235294117647, 1.7, 0.649982166210914])
    # print("Test parameters are used")
    
    for tauInd in range(len(plotTauLags)):
        ax.plot(ksq2plot, fit_fun(opt_params,ksq2plot,plotTauLags[tauInd]))
    # tightfig(gfc)?
    
    # saving
    filename = ('ROI_X_'+str(roi[i,0])
                +'-'+str(roi[i,0]+roi[i,2]-1)
                +'_Y_'+str(roi[i,1])
                +'_'+str(roi[i,1]+roi[i,3]-1)
                )
    
    print("Optimal parameter are: " + str(opt_params))
    # d = {'r_k_norm': r_k_norm, 'kSqMin': kSqMin, 'kSqMax': kSqMax,
    #      'opt_params': opt_params, 'win_fit': win_fit, 'time_win': time_win,
    #      'tauVector': tauVector, 'err_min': err_min}

    # with open(filename, 'wb') as f:
    #     pickle.dump(d, f)

