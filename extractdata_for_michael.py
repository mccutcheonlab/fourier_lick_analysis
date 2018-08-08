# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 05:22:28 2018

@author: jaimeHP
"""
import numpy as np

import JM_general_functions as jmf
import csv


medfolder = 'R:\\DA_and_Reward\\kp259\\DPCP3\\med\\'
filename = '!2017-11-25_09h48m.Subject dpcp3.08'
file = medfolder + filename

onset, offset = jmf.medfilereader(file,
                                  varsToExtract=['e', 'f'],
                                  remove_var_header=True)

with open(filename + '.csv', 'w+', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Onset', 'Offset'])
    for row in np.transpose([onset, offset]):
        w.writerow(row)
        
        
#f = open('testlicks.csv')