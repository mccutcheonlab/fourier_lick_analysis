# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 13:23:57 2018

@author: jaimeHP
"""
import numpy as np

import JM_general_functions as jmf
import matplotlib.pyplot as plt

from statsmodels.nonparametric.smoothers_lowess import lowess

def smoothed_session_plot(ax, y, x=[]):
    
    if len(x) != len(y):
        x = range(0,len(y))
        
    smoothed_y = lowess(y, x, is_sorted=True, frac=0.025, it=0)
    smoothed_y = np.transpose(smoothed_y)
    
    ax.plot(smoothed_y[1])
    
    return ax

def shuffle_licks(licks):
    ilis = np.diff(np.concatenate([[0], licks]))   
    shuffled_ilis = np.random.permutation(ilis)
    shuffled_licks = np.cumsum(shuffled_ilis)
    
    return shuffled_licks

# get med data file(s)
medfolder = 'C:\\Users\\jaimeHP\\Documents\\GitHub\\fourier_lick_analysis\\'
filename = medfolder + '!2017-10-06_09h31m.Subject dpcp2.10'

onset, offset = jmf.medfilereader(filename,
                                  varsToExtract=['e', 'f'],
                                  remove_var_header=True)

binned_licks = np.histogram(onset, bins=3600, range=(0,3600))
binned_shuffled_licks = np.histogram(shuffle_licks(onset), bins=3600, range=(0,3600))


#t, y = jmf.discrete2continuous(onset, offset, fs=10)



#ax.plot(y)

#smoothed_licks = lowess(binned_licks[0], binned_licks[1][1:], is_sorted=True, frac=0.025, it=0)
#smoothed_licks = np.transpose(smoothed_licks)

#ax.plot(smoothed_licks[1])


fig1, ax = plt.subplots(nrows=2)
smoothed_session_plot(ax[0], binned_licks[0])
smoothed_session_plot(ax[1], binned_shuffled_licks[0])

