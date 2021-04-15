#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:48:29 2021

@author: nmlock
"""


import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import pickle
from tds_remove_correlation import *
plt.close('all')

"""
load profiles
"""

test = pickle.load( open ('profiles_feb_phase1.pkl', "rb"))
test2 = pickle.load(open('profiles_feb_phase2.pkl', "rb"))


chirps1 = test.index.to_numpy()
data1 = test.to_numpy()
chirps2 = test2.index.to_numpy()
data2 = test2.to_numpy()


i = 7
plt.figure('profile')
plt.plot(data1[i])
plt.plot(data2[i])


x1= np.arange(np.size(data1[i]))
x2= np.arange(np.size(data2[i]))

"""
Preparation for tomography: both profiles with same time scale, centering, normalization
"""

shear_para =0.0036447  #m/ps assume some random shear parameter from the phase scan: pixel to time
pixel_size = 5.4e-6 # onepixel -> 5.4 um
time1 = x1*pixel_size/shear_para*1e-12 # in s
time2 = x2*pixel_size/shear_para*1e-12 # in s
#bring both profiles on same time axis
if np.size(time1) > np.size(time2):
    time = time1
else:
    time = time2
profiles = np.zeros([2,np.size(time)])
#centering
time = time-np.mean(time)
time1 = time1-first_moment(data1[i], time1)
time2 = time2-first_moment(data2[i], time2)

profiles[0] = np.interp(time, time1, data1[i])
profiles[1] = np.interp(time, time2, data2[i])

#normalization
profiles[0] = profiles[0] / np.trapz(profiles[0], time)
profiles[1] = profiles[1] / np.trapz(profiles[1], time)

plt.figure('Input to 2-point')
plt.plot(time, profiles[0])
plt.plot(time, profiles[1])



"""
Test function
"""
recon, tilt = two_point_tomo(profiles[0], profiles[1], time, show_plots = False)

plt.figure('Final comparison')
plt.plot(time*1e15, profiles[0], label = 'phase 1')
plt.plot(time*1e15, profiles[1][::-1], label = 'phase 2')
plt.plot(time*1e15, recon, label = 'recon')
plt.xlabel('time (fs)')
plt.ylabel('norm. current (1/fs)')
plt.legend()