#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 11:28:56 2021

@author: lockmann
"""

import numpy as np 
from scipy.io import loadmat, savemat
from os import walk, getcwd
import re
import matplotlib.pylab as plt

mypath =getcwd()
filenames = next(walk(mypath), (None, None, []))[2]  # [] if no file
plt.figure()
for file in filenames:
    if re.match(r'formfactor.*\.mat',file):
        print (file)
        crisp_file = file
        crisp_data =loadmat(crisp_file,squeeze_me = True)
        crisp_f = crisp_data['frequencies']
        crisp_ff = crisp_data['formfactors']
        crisp_ff_noise = crisp_data['formfactors_err']
        crisp_detlim = crisp_data['det_limit']
        #crisp response anpassen 
        #apply newer response correction
        tds_cor_old = 'tds_response_correction.mat'
        tds_cor_new = 'response_correction_diss.mat'
        #plt.figure()
        old_resp_cor = loadmat(tds_cor_old,squeeze_me = True)['tds_resp_corr'][::-1]
        new_resp_cor = loadmat(tds_cor_new,squeeze_me = True)['resp_corr'][::-1]
        #und noch korrigieren wegen neuer modelled response
        new_resp_cr = loadmat('dr5mm_100ns_sensitivity_new.mat', squeeze_me =True)['sensitivity'] [::-1] #*1.55  
        old_resp_cr = loadmat('dr5mm_100ns_sensitivity.mat', squeeze_me =True)['sensitivity'] [::-1] *2 #*1.55*2
        old_resp = old_resp_cr /old_resp_cor
        new_resp = new_resp_cr /new_resp_cor
        #new_resp_corr = 1
        
        #Disapply old response: 
        crisp_ff = crisp_ff*np.sqrt(old_resp) / np.sqrt(new_resp)
        crisp_ff_noise=  crisp_ff_noise*np.sqrt(old_resp) / np.sqrt(new_resp)                                     
        crisp_detlim = crisp_detlim *np.sqrt(old_resp) / np.sqrt(new_resp)
        charge = crisp_data['charge']*1e-9
        # das detlim ist komisch. nehme das von den adneren messungen und passe es einfach auf die anahl der schüsse -> 50, und die ladung an
        crisp_detlim =loadmat('2018-10-30_l1_113.mat',squeeze_me = True)['det_limit'][::-1]
        crisp_detlim = crisp_detlim*0.25e-9/charge # doppelwurzel weil f nicht f²
        crisp_detlim = crisp_detlim*20**(-1./4.)*100**(1./4.) # weil hier 20 und vorher 50 schuesse
        plt.loglog(crisp_f*1e-12, crisp_ff)
        
        my_dict = {'frequencies':crisp_f, 'formfactors':crisp_ff, 'formfactors_err': crisp_ff_noise,
                   'det_lim':crisp_detlim, 'charge':charge}
        savemat(crisp_file.replace('formfactor', 'formfactor_new'), my_dict)