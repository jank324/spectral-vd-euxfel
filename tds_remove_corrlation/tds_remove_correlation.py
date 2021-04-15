#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:51:03 2020

Two-Point Tomographie from files from tds_image_to_profiles.py. Get the 
beam tilt from the average profile and than apply the beam tilt to the 
single profiles. 

@author: lockmann
"""

import numpy as np
import matplotlib.pylab as plt
from scipy.io import loadmat,savemat
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter



def first_moment(intensity, x):
    return np.sum(intensity*x)/np.sum(intensity)

def second_moment(intensity, x):
    return np.sum(intensity*(x-first_moment(intensity, x))**2)/np.sum(intensity)

def two_point_tomo(profile1, profile2, time, show_plots = True):
    """
    Profiles need to be the same time scale, and the direction must be as on the screen, i.e.
    no correction of a negative shear parameter applied before
    
    Returns: two arrays (recon, tilt) with respect to input time
        recon: current profile obtained by tomographie of both phases
        tilt: beam tile intrinsic x coordniate along z
    """

    if show_plots == True:
        plt.figure()
        plt.plot(time*1e15, profile1, label = 'profile 1')
        plt.plot(time*1e15, profile2, label = 'profile 2')
        plt.xlabel('t')
        plt.ylabel('I (a.u.)')

    #normieren
    mu1= profile1/np.trapz(profile1, x = time)
    mu2 = profile2 / np.trapz(profile2, x= time)

    #estimate time window sufficiently large -> 10 Sigma
    width1 = 5*np.sqrt(second_moment(profile1, time))
    width2 = 5*np.sqrt(second_moment(profile2,time))
    width = np.amax([width1,width2])
    
    #Particle Fraction
    q_pos =cumtrapz(mu1,x = time)
    q_neg = cumtrapz(mu2[::-1], x=time)#,x = time_axis_neg)
    q_neg = q_neg[::-1]
    new_time= (time[1:] + time[:-1]) / 2.
    if show_plots == True:
        plt.figure('Particle Fraction')
        plt.plot(new_time*1e15, q_pos,'o')
        plt.plot(new_time*1e15,q_neg,'o')

    #Inverse functions y
    inter_y_pos = interp1d(q_pos, new_time, kind = 'linear')#, bounds_error = False, fill_value = (0,1)) # x und y vertausch
    inter_y_neg = interp1d(q_neg, -new_time, kind = 'linear')#,bounds_error=False, fill_value = (1,0))
    #There is a lot of uninteresting data. 
    q_low = np.amax([np.amin(q_pos), np.amin(q_neg)])+1e-4 # to have some margin before going intp bounds error
    q_high = np.amin([np.amax(q_pos), np.amax(q_neg)])-1e-4
    q_data = np.linspace(q_low,q_high,2**8)
    y_pos = inter_y_pos(q_data)
    y_neg = inter_y_neg(q_data)
    
    if show_plots == True:
        plt.figure('Inverse Function y(q)')
        plt.plot(q_pos, new_time,'o')
        plt.plot(q_neg, -new_time,'o')
        plt.plot(q_data, y_pos, label = 'u1')
        plt.plot(q_data, y_neg, label = 'u2')

    #Differenez davon 
    z = (y_pos+y_neg)/2.
    if show_plots == True:
        plt.plot(q_data, z, '-o', label = 'z(q)')
        plt.legend()
    
    #Inverse function F from u
    inter_q= interp1d(z, q_data,  fill_value = (0,1), bounds_error=False)
    z_lim = np.amax([-np.amin(z),np.amax(z)])
    z_data = np.linspace(-width,width,2**8)
    q_inter = inter_q(z_data)
    
    if show_plots == True:
        plt.figure('F (z)')
        plt.plot(z_data,q_inter)
        plt.plot(z,q_data,'o')

    """
    The deriviation is the bunch profile
    """
    lambda_from_z = np.diff(q_inter)/np.diff(z_data)
    #Subtract BKR -> this needs than off course renormalization
    lambda_from_z[lambda_from_z<np.amax(lambda_from_z)/100] = 0
    new_z = (z_data[1:] + z_data[:-1]) / 2
    #renormalize
    lambda_from_z = lambda_from_z / np.trapz(lambda_from_z, x= new_z)
    new_z = new_z -  first_moment(lambda_from_z, new_z)
    

    """
    smooth reconstruction data to remove steps
    """
    lambda_from_z_smooth = savgol_filter(lambda_from_z, 11, 2) # window size 51, polynomial order 3
    lambda_from_z_smooth[lambda_from_z_smooth<0] = 0
    
    sigma = np.sqrt(second_moment(lambda_from_z_smooth, new_z))
    
    if show_plots == True:
        plt.figure('Rekonstruction')
        plt.plot(new_z*1e15, lambda_from_z_smooth, label = r'$\sigma_t=$ ' + str(np.round(sigma*1e15,1)) + ' fs')
        plt.xlim([-7*sigma*1e15,7*sigma*1e15])
        plt.xlabel('fs')
        plt.legend()
    
    """
    The Beam tilt is 
    """
    f0 = y_neg - y_pos
    if show_plots == True:
        plt.figure('Beam tilt')
        plt.title('Beam tilt')
        plt.plot(z*1e15,f0*1e15)
        plt.xlim([-7*sigma*1e15,7*sigma*1e15])
        plt.xlabel('Intra-Bunch Coordinate t (fs)')
        plt.ylabel(r'$x / S_x$ (fs)')
    
    recon_current = interp1d(new_z, lambda_from_z_smooth,fill_value = (0,0), bounds_error=False) (time)
    beam_tilt = interp1d(z, f0,fill_value = (0,0), bounds_error=False) (time)
    
    
    return recon_current, beam_tilt



