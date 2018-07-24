# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 11:30:26 2018

@author: jaimeHP
"""
import numpy as np
import time
import matplotlib.pyplot as plt

from statsmodels.nonparametric.smoothers_lowess import lowess

def medfilereader(filename, varsToExtract = 'all',
                  sessionToExtract = 1,
                  verbose = False,
                  remove_var_header = False):
    if varsToExtract == 'all':
        numVarsToExtract = np.arange(0,26)
    else:
        numVarsToExtract = [ord(x)-97 for x in varsToExtract]

    f = open(filename, 'r')
    f.seek(0)
    filerows = f.readlines()[8:]
    datarows = [isnumeric(x) for x in filerows]
    matches = [i for i,x in enumerate(datarows) if x == 0.3]
    if sessionToExtract > len(matches):
        print('Session ' + str(sessionToExtract) + ' does not exist.')
    if verbose == True:
        print('There are ' + str(len(matches)) + ' sessions in ' + filename)
        print('Analyzing session ' + str(sessionToExtract))

    varstart = matches[sessionToExtract - 1]
    medvars = [[] for n in range(26)]

    k = int(varstart + 27)
    for i in range(26):
        medvarsN = int(datarows[varstart + i + 1])

        medvars[i] = datarows[k:k + int(medvarsN)]
        k = k + medvarsN

    if remove_var_header == True:
        varsToReturn = [medvars[i][1:] for i in numVarsToExtract]
    else:
        varsToReturn = [medvars[i] for i in numVarsToExtract]

    if np.shape(varsToReturn)[0] == 1:
        varsToReturn = varsToReturn[0]
    return varsToReturn

def isnumeric(s):
    try:
        x = float(s)
        return x
    except ValueError:
        return float('nan')

def discrete2continuous(onset, offset=[], nSamples=[], fs=[]):
    # this function takes timestamp data (e.g. licks) that can include offsets
    # as well as onsets, and returns a digital on/off array (y) as well as the
    # x output. The number of samples (nSamples) and sample frequency (fs) can
    # be input or if they are not (default) it will attempt to calculate them
    # based on the timestamp data. It has not been fully stress-tested yet.

    try:
        fs = int(fs)
    except TypeError:
        isis = np.diff(onset)
        fs = int(1 / (min(isis)/2))

    if len(nSamples) == 0:
        nSamples = int(fs*max(onset))

    outputx = np.linspace(0, nSamples/fs, nSamples)
    outputy = np.zeros(len(outputx))

    if len(offset) == 0:
        for on in onset:
            idx = (np.abs(outputx - on)).argmin()
            outputy[idx] = 1
    else:
        for i, on in enumerate(onset):
            start = (np.abs(outputx - on)).argmin()
            stop = (np.abs(outputx - offset[i])).argmin()
            outputy[start:stop] = 1

    return outputx, outputy

def smoothed_session_plot(ax, y, x=[]):

    if len(x) != len(y):
        x = range(0,len(y))

    smoothed_y = lowess(y, x, is_sorted=True, frac=0.025, it=0)
    smoothed_y = np.transpose(smoothed_y)

    ax.plot(smoothed_y[1])

    return ax

#def savitzky_golay(y, window_size, order, deriv=0, rate=1):
#    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
#    The Savitzky-Golay filter removes high frequency noise from data.
#    It has the advantage of preserving the original shape and
#    features of the signal better than other types of filtering
#    approaches, such as moving averages techniques.
#    Parameters
#    ----------
#    y : array_like, shape (N,)
#        the values of the time history of the signal.
#    window_size : int
#        the length of the window. Must be an odd integer number.
#    order : int
#        the order of the polynomial used in the filtering.
#        Must be less then `window_size` - 1.
#    deriv: int
#        the order of the derivative to compute (default = 0 means only smoothing)
#    Returns
#    -------
#    ys : ndarray, shape (N)
#        the smoothed signal (or it's n-th derivative).
#    Notes
#    -----
#    The Savitzky-Golay is a type of low-pass filter, particularly
#    suited for smoothing noisy data. The main idea behind this
#    approach is to make for each point a least-square fit with a
#    polynomial of high order over a odd-sized window centered at
#    the point.
#    Examples
#    --------
#    t = np.linspace(-4, 4, 500)
#    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
#    ysg = savitzky_golay(y, window_size=31, order=4)
#    import matplotlib.pyplot as plt
#    plt.plot(t, y, label='Noisy signal')
#    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
#    plt.plot(t, ysg, 'r', label='Filtered signal')
#    plt.legend()
#    plt.show()
#    References
#    ----------
#    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
#       Data by Simplified Least Squares Procedures. Analytical
#       Chemistry, 1964, 36 (8), pp 1627-1639.
#    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
#       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
#       Cambridge University Press ISBN-13: 9780521880688
#    """
#    import numpy as np
#    from math import factorial
#    
#    try:
#        window_size = np.abs(np.int(window_size))
#        order = np.abs(np.int(order))
#    except ValueError, msg:
#        raise ValueError("window_size and order have to be of type int")
#    if window_size % 2 != 1 or window_size < 1:
#        raise TypeError("window_size size must be a positive odd number")
#    if window_size < order + 2:
#        raise TypeError("window_size is too small for the polynomials order")
#    order_range = range(order+1)
#    half_window = (window_size -1) // 2
#    # precompute coefficients
#    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
#    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
#    # pad the signal at the extremes with
#    # values taken from the signal itself
#    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
#    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
#    y = np.concatenate((firstvals, y, lastvals))
#    return np.convolve( m[::-1], y, mode='valid')

rfolder = '/rfs/DA_and_Reward/kp259/DPCP2/'
medfolder = '!2017-10-06_09h31m.Subject dpcp2.10'

rfolder = 'R:\\DA_and_Reward\\kp259\\DPCP2\\'
medfolder = '!2017-10-06_09h31m.Subject dpcp2.10'

filename = rfolder+medfolder
print(filename)

onset, offset = medfilereader(filename, varsToExtract=['e', 'f'],
                               remove_var_header=True)

binned_licks = np.histogram(onset, bins=3600, range=(0,3600))

tic = time.clock()
t, y = discrete2continuous(onset, offset, fs=10)

Y = np.fft.fft(y)
freq = np.fft.fftfreq(len(Y), t[1]-t[0])
toc = time.clock()
print(toc-tic)


fig1, ax = plt.subplots(nrows=2)
smoothed_session_plot(ax[0], binned_licks[0])
ax[1].set_xscale('log')
ax[1].set_yscale('log')
#ax[1].plot(freq, abs(Y))
smoothed_session_plot(ax[1], abs(Y))

