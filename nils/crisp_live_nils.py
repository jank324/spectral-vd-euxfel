#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 14:32:53 2021

@author: lockmann
"""

# import pydoocs
import time
import numpy as np
import matplotlib.pylab as plt
from nils.reconstruction_module import master_recon

plt.close('all')


def get_real_crisp_data(shots = 20, which_set = 'both'):
    # load current crisp data
    crisp_adress = 'XFEL.SDIAG/THZ_SPECTROMETER.FORMFACTOR/CRD.1934.TL/'
    
    """
    Collect Data
    """
    #shots = 20
    all_data = np.empty([shots, 240])
    freqs = pydoocs.read(crisp_adress+ 'FORMFACTOR.XY')['data'] [:,0]
    for shot in np.arange(shots):
        t0 = time.time()
        all_data[shot]= pydoocs.read(crisp_adress + 'FORMFACTOR.XY')['data'][:,1]
        t1 = time.time()
        dt = t1-t0
        if dt<0.1:
            time.sleep(0.1-dt)
    
    #get detection limit calculated by sigma of 10? rf pulses
    det_lim =  pydoocs.read(crisp_adress + 'FORMFACTOR_MEAN_DETECTLIMIT.XY')['data'][:,1]
    
    """
    Average and noise and make comparable to simulated data
    """
    freqs_to_return = freqs*1e12
    det_lim = np.sqrt ( det_lim * np.sqrt(10/shots)) # (1 sigma in electronic noise?! und auf richtige schussanzahl anpassen
    
    ff_sq_mean = np.mean(all_data, axis=0)
    ff_to_return = np.sqrt(np.abs(ff_sq_mean)) * np.sign(ff_sq_mean) 
    ff_sq_noise = np.std(all_data, axis = 0)
    ff_noise = np.abs(0.5/ff_to_return * ff_sq_noise)
    if which_set == 'high':
        freqs_to_return = freqs_to_return[120:]
        ff_to_return = ff_to_return[120:]
        ff_noise = ff_noise[120:]
        det_lim = det_lim[120:]
    elif which_set == 'low':
        freqs_to_return = freqs_to_return[:120]
        ff_to_return = ff_to_return[:120]
        ff_noise = ff_noise[:120]
        det_lim = det_lim[:120]
    #plt.figure()
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.errorbar(freqs_to_return*1e-12, ff_to_return, yerr = ff_noise)
    #plt.fill_between(freqs_to_return *1e-12, np.zeros(240), y2 = det_lim, color = 'gray', alpha = 0.3)
    

    return np.array([freqs_to_return, ff_to_return, ff_noise, det_lim])

def get_charge(shots = 20):
    tor_adresse = 'XFEL.SDIAG/THZ_SPECTROMETER.FORMFACTOR/CRD.1934.TL/CHARGE.TD'
    test = pydoocs.read(tor_adresse)['data'][:,1]
    all_data = np.empty([shots,np.size(test)])
    for shot in np.arange(shots):
        t0 = time.time()
        all_data[shot]= pydoocs.read(tor_adresse)['data'][:,1]
        t1 = time.time()
        dt = t1-t0
        if dt<0.1:
            time.sleep(0.1-dt)
    charge = np.mean(all_data, axis=0)[0]*1e-9
    return charge


if __name__ == "__main__":
    frequency, formfactor, formfactor_noise, detlim = get_real_crisp_data(20, "high")
    charge = get_charge()
    t, current, _ = master_recon(frequency, formfactor, formfactor_noise, detlim, charge,
                                 method="KKstart", channels_to_remove=[], show_plots=True)
