This is an translation of the code from: https://github.com/ssehayek/kics-project into python.

# How to use this kICS code:

1. Download modules
2. Have some diffusing particle data that is in need of analysing.
3. Open the file kicsSimAnalysis.py
4. Input the filepath+filename under the variable filename
5. Change parameters to liking
6. Run the programm

Different parameters:
use_time_win: Logical value for using a time window on the autocorrelation.
time_win: If a time window is used on the autocorrelation input the size in frames of the time window here. Recommended is a time window of 200 frames.
win_fit: Logical value for choosing between the two fitting functions. False for the standard fitting function that doesn't take the time window into account. True for the time-window correction model. Recommended is to only use the time window correction model, when the standard model doesn't give a good fit.
Save: Logcial value if the graphic should be saved automatically.
tauVector: Range of time lags that will be included in the fitting. 
plotTauLags: Range of time lags shown in the diagramm
kSqMin, kSqMax: Bounds to |k|^2. Recommended is between close to 0 and 1.8.
Roi: Region of interest to analyse. This should be a vector with one ROI per row in the form ([x0,y0,width,height]), with x0 and y0 being the start coordinates and width and height the width and height of the ROI in pixels.
ksq_min_noise, ksq_max_noise: Bound on |k|^2 with respect to the noise.
n_fits: Number of fits that will be compared to combat the random start of the fitting.
nPtsFitPlot: Resolution of the fitting curve.
Bounds: This are the lower/upper bounds of the fit parameters ([D,rho_on,K,p_D]) this can be changed to influence the fitting.

# License
[MIT](LICENSE) © Martin Dimler

