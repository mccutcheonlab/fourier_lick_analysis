# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 13:00:36 2018

@author: jaimeHP
"""

import numpy as np

import matplotlib.pyplot as plt

varsin = np.load('session_save.npz')

fig1, ax = plt.subplots(nrows=2)


ax[0].set_xscale('log')
ax[0].set_yscale('log')

#x = [x for x in freq if x>0]

ax[0].plot(varsin['x'], varsin['Y'])