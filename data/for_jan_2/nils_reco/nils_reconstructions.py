#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 15:00:32 2022

@author: nmlock
"""


import numpy as np
import matplotlib.pylab as plt
from scipy.io import loadmat, savemat
from reconstruction_module_after_diss import *

plt.close('all')

"""
FOR L1
"""
files = ['formfactor_new_14_28_25_l1_-10.0.mat', 'formfactor_new_14_28_25_l1_-10.5.mat', 
         'formfactor_new_14_28_25_l1_-11.0.mat','formfactor_new_14_28_25_l1_-11.5.mat',
         'formfactor_new_14_28_25_l1_-12.0.mat']
tds_data = loadmat('../TDS/tds_l1_results.mat', squeeze_me = True)
tds_indices = [-1,-2,-3,-4,-5]

"""
FOR L2
"""
files = ['formfactor_new_11_59_57_l2_6.0.mat', 'formfactor_new_11_59_57_l2_4.0.mat', 
         'formfactor_new_11_59_57_l2_-2.0.mat','formfactor_new_11_59_57_l2_-4.0.mat',
         'formfactor_new_11_59_57_l2_-6.0.mat','formfactor_new_11_59_57_l2_-9.0.mat',
         'formfactor_new_11_59_57_l2_-10.0.mat','formfactor_new_11_59_57_l2_-11.0.mat',
         'formfactor_new_11_59_57_l2_-12.0.mat', 'formfactor_new_11_59_57_l2_-13.0.mat']
tds_data = loadmat('../TDS/tds_l2_results.mat', squeeze_me = True)
tds_indices = [-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]

for i in np.arange(np.size(files)):
    #load crisp
    data = loadmat('../CRISP/'+ files[i], squeeze_me = True)
    #recon crisp
    recon_time, current, t_rms, new_freqs, recon_ff, cleandata =  master_recon(data['frequencies'], data['formfactors'], data['formfactors_err'], data['det_lim'], data['charge'],show_plots = True)
    #load tds
    tds_time = tds_data['time']
    tds_current = tds_data['currents'][tds_indices[i]]
    
    #plot vergleich tds <-> Crisp reconstruction
    plt.figure(files[i])
    plt.title(files[i])
    plt.plot(recon_time*1e15, current*1e-3, label = 'CRISP')
    plt.plot(tds_time*1e15, tds_current*1e-3, label = 'TDS')
    plt.legend()
    plt.xlim([-300,300])
    
    #speicher bild und daten
    np.savetxt(files[i].replace('.mat', '_recon.txt'), np.transpose(np.array([recon_time*1e15, current*1e-3])))
    
    plt.savefig(files[i].replace('.mat', '.png'))