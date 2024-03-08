# -*- coding: utf-8 -*-
"""This module contains code for analysing a image series of diffusing particles with the extended kICS method.
Included are two fitting functions: one which takes the time window into account and one who doesn't. The autocorrelation,
The corresponding fits and the parameters are outputted.

Created on Thu Jan 12 11:44:41 2023
@author: Martin
"""
# Import of packages
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import cv2

# Import of functions from other modules.
from kics_kics import kICS
from kics_kicsSubNoise import kICSSubNoise
from kics_circular import circular
from kics_getKSqVector import getKSqVector
from kics_kicsSlideWinFit import kICSSlideWinFit
from kics_timeIntkICsFit import timeIntkICSFit
#%%

# input variables
# Filepath of image series that is going to be analysed
filename = './SimulationBleach/frac_diff_1_2_0000001bleach.tif'

# logical if time window will be included in analysis and size of the time window in frames.
use_time_win = 0
time_win = 200

# logical for including windowing in fit model
win_fit = 0

# logical if graphic should be saved
save = 0

# tau range to fit
tauVector = np.arange(10)
# tau range to plot
plotTauLags = np.arange(5)

# min/max bounds on |k|^2
kSqMin = np.spacing(1)
kSqMax = 1.8

#  ROIs to consider (one per row; form [x0,y0,width,height])
if np.size(J,1) >= 128 and np.size(J,0) >= 128:
    roi = np.array([[0,0,63,63],
            [64,64,63,63],
            [0,64,63,63],
            [64,0,63,63]
            ])
else:
    roi = np.array([[0,0,31,31],
            [32,32,31,31],
            [0,32,31,31],
            [32,0,31,31]
            ])

# roi = np.array([[0,0,127,127]])

# min/max bounds on |k|^2
ksq_min_noise = 10
ksq_max_noise = 15

# number of fits to compare
n_fits = 5

# Number of points the fit is plotted on
nPtsFitPlot = int(1e3)

# lower/upper bounds on fit parameters, [D,rho_on,K,p_D]
bounds = [(np.spacing(1),np.inf), # np.inf
          (np.spacing(1),1),
          (np.spacing(1),np.inf), # np.inf
          (np.spacing(1),1),
          ]

# import of images through the filename
im = cv2.imreadmulti(filename,[],cv2.IMREAD_UNCHANGED)
J = np.asarray(im[1],dtype = np.float64)
J = np.moveaxis(J,0,-1)
print(np.shape(J))
T = np.size(J,2)


#%%
# main code

for i in range(np.size(roi,0)):
    # ROI of the image series
    J_roi = np.copy(J[roi[i, 1]:roi[i, 1]+roi[0, 3]+1,
                    roi[i, 0]:roi[i, 0]+roi[i, 2]+1, :])

    # compute kICS autocorr

    #kICS autocorrelation function (ACF)
    if use_time_win:
        time_win_varargin = ('timeWin', time_win)
    else:
        time_win_varargin = ()
    
    r_k = kICS(J_roi, ('normByLag', 'none', *time_win_varargin))
    
    r_k_0_sub = kICSSubNoise(r_k, ksq_min_noise, ksq_max_noise)
    
    # Circular Averaging
    r_k_0_circ,_ = circular(r_k_0_sub[:,:])
    r_k_circ,_ = circular(r_k)
    # get and cut |k|^2 vector
    # Needed to plot kICS autocorrelation at tau=0
    kSqVectorSubset, kSqSubsetInd = getKSqVector(
        J_roi, ('kSqMin', kSqMin, 'kSqMax', kSqMax))

    # cut autocorrelation
    r_k_circ_cut = np.array([r_k_circ[j][tauVector+1] for j in kSqSubsetInd])
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
        ax.scatter(kSqVectorSubset,r_k_norm[:,plot_idx[tauInd]], label=r'$\tau = \ %d $' %((plotTauLags[tauInd]+1)), s=10)

    ax.set_xlabel(r'$|k|^2$ (pixels$^{-2}$)')
    ax.set_ylabel(r'$\tilde{\Phi}(|$k$|^2,\tau)$')
    ax.legend()
    ax.set_xlim([kSqVectorSubset[0], kSqVectorSubset[-1]])
    # ax.set_ylim()

    
    for tauInd in range(len(plotTauLags)):
        ax.plot(ksq2plot, fit_fun(opt_params,ksq2plot,plotTauLags[tauInd]))
    
    # saving
    if save:
          filename = ('ROI_X_'+str(roi[i,0])
                          +'-'+str(roi[i,0]+roi[i,2]-1)
                          +'_Y_'+str(roi[i,1])
                          +'_'+str(roi[i,1]+roi[i,3]-1)
                          )
    
    print("Optimal parameter are: ")
    for parm in opt_params:
        print(parm)
