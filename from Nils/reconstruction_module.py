# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:11:54 2019

Module for the different tasks that need to be done to reconstruct the current 
profile from a formfactor measurement

@author: lockmann
"""

import numpy as np
import matplotlib.pylab as plt
from scipy.io import loadmat
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.signal import savgol_filter

"""
General Helper Functions
"""
def first_moment(intensity, x):
    return np.sum(intensity*x)/np.sum(intensity)

def second_moment(intensity, x):
    return np.sum(intensity*(x-first_moment(intensity, x))**2)/np.sum(intensity)

def super_gauss(x,amp,order,sigma):
    """
    Returns a super gaussian function with the given parameters
    """
    return amp *np.exp(-0.5*(x/sigma)**(2*int(order)))

def gauss(x, sigma):
    """
    Gauss Function with amplitude 1
    """
    return super_gauss(x,1,1,sigma)

def gauss_wamp(x, amp,sigma):
    """
    Gauss Function with variable amplitude bigger than 1
    """
    if amp<1:
        amp =1
    return amp*super_gauss(x,1,1,sigma)

def nils_fft(xin, datain):
    """
    Function to return 
    """
    fouriertransform = np.fft.fft(datain)
    quatschfaktor = 1/np.sqrt(2. *np.pi) * (xin[1]-xin[0])
    fouriertransform = quatschfaktor*fouriertransform
    fx = np.fft.fftfreq(np.size(xin), d = xin[1]-xin[0])
    #fx = sort(fx)
    wx = 2*np.pi*fx
    return wx, fouriertransform    


"""
Functions for processing of CRISP data
"""

def remove_certain_channels(channel_list, dataset):
    """
    Remove certain channels from the formfactor (broken ones for example)
    """
    new_data = np.copy(dataset)
    new_data = np.delete(new_data, channel_list, axis = 1)
    return new_data
    

def sort_data(frequencies, data_to_sort):
    """
    Sort data according to frequency. Very important for everything to come. 
    """
    if np.size(np.shape(data_to_sort))>1:
         result = np.empty_like(data_to_sort)
         for i in np.arange(np.shape(data_to_sort)[0]):
             this_sorted = data_to_sort[i][np.argsort(frequencies)]
             result[i] = this_sorted
    return np.sort(frequencies), result
             
def remove_loners(sorted_frequencies,sorted_formfactors, sorted_noise, sorted_det_limit):
    """
    we don't want single channels to pop up in regions where there is not much signal. 
    Therefore, we demand 4 neighbours above the significanc level and must not be nan
    """
    my_maske = np.ones(np.size(sorted_frequencies), dtype = 'bool')
    for channel in np.arange(np.size(sorted_formfactors)):
        this_values = sorted_formfactors[channel-2:channel+2]
        this_det_limits = sorted_det_limit[channel-2:channel+2]
        if np.any(this_values<this_det_limits) or np.any(np.isnan(this_values)):
            my_maske[channel] = False
        #edit 21.03.2020: alle unter dem detection limit, sind auch kleiner als das detection limit aber mit unendlich großem fehler
    new_sorted_formfactors = np.copy(sorted_formfactors)
    new_sorted_noise = np.copy(sorted_noise)
    new_sorted_formfactors[~my_maske] = 0.5*sorted_det_limit[~my_maske]
    new_sorted_noise[~my_maske] = 0.5* sorted_det_limit[~my_maske]
    return sorted_frequencies[my_maske], sorted_formfactors[my_maske], sorted_noise[my_maske]
    #return sorted_frequencies, new_sorted_formfactors, new_sorted_noise
        
#def remove_outliers(sorted_formfactors, n_neighbours = 3):
#    """
#    Check for unphysical jumps, if the values change more than 0.5/2 in the 
#    neighbours range, the data point is removed
#    """
#    channels = np.arange(np.size(sorted_formfactors))[::-1]
#    changes= sorted_formfactors[1:]/sorted_formfactors[:-1]
#    low_jumps = channels[changes<0.5]
#    high_jumps = channels[changes>2.]
#    distances = n

def extrapolation_low(new_frequencies, sorted_frequencies, sorted_formfactors, sorted_formfactor_noise):
    """
    Extrapolation to low Frequencies. As soon as 3 channels have a formfactor bigger than ff_value,
    A Gaussian function will be fitted and this data replaced by the fit. 
    Question: Allow Formfactors bigger than 1? -> No
    """
    
    #find the value above which it will be extrapolated
    ff_value = 0.2
    while ff_value<0.9 and np.size(sorted_formfactors[sorted_formfactors>ff_value])>8:
        ff_value = ff_value +0.1
        
    
    channels = np.arange(np.size(sorted_frequencies))
    #find where the first time 3 ff are smaller than ff_value
    smallff_channels = channels[sorted_formfactors<ff_value]
    #scan to fnd the first time 3 smaller 0.8
    n_hits = 0
    i =1
    while n_hits <2:
        if smallff_channels[i] == smallff_channels[i-1]+1:
            #print(smallff_channels[i])
            n_hits = n_hits +1
        else:
            n_hits=0
        #print(n_hits)
        i=i+1
    n_channels_tmp =smallff_channels[i]
    # Long Bunches: 
    if n_channels_tmp <4:
        n_channels = 60
    else:
        n_channels = n_channels_tmp
    #print('stopped at ' + str(sorted_frequencies[n_channels]*1e-12))
    #n_channels = np.amax(channels[sorted_formfactors>0.8])
    fit_freqs, fit_formfactors = sorted_frequencies[:n_channels], sorted_formfactors[:n_channels]
    guess = 1e12    
    popt, pcov = curve_fit(gauss, fit_freqs, fit_formfactors, p0 = guess) 
    #guess_width = np.sqrt(-sorted_frequencies[n_channels]**2/ 2 / np.log(sorted_formfactors[n_channels]))
    #guess = np.array([1,guess_width])
    #popt, pcov = curve_fit(gauss_wamp, fit_freqs, fit_formfactors, p0 = guess) 
    #for long bunches to only extrapolate but not replace:
    if n_channels_tmp<4:
        n_channels = 0
    added_freqs = new_frequencies[new_frequencies<sorted_frequencies[n_channels]]
    added_formfactor = gauss(added_freqs, popt)#[0], popt[1])
    new_formfactors = np.append(added_formfactor, sorted_formfactors[n_channels:])
    new_noise = np.append(np.zeros_like(added_formfactor), sorted_formfactor_noise[n_channels:])
    new_freqs = np.append(added_freqs, sorted_frequencies[n_channels:])
    return new_freqs, new_formfactors, new_noise

def extrapolation_high(new_frequencies, sorted_frequencies, sorted_formfactors, sorted_formfactor_noise, index_to_cut = 1):
    """
    Extrapolation to high Frequencies. A Gaussian will be fitted to the value
    at the data point at the highest available frequency. 
    """
    last_freq = sorted_frequencies[-index_to_cut]
    last_mess = sorted_formfactors[-index_to_cut]
    high_sig_f = np.sqrt(-last_freq**2/ 2 / np.log(last_mess))
    
    fit_freqs= new_frequencies[new_frequencies>last_freq]
    fit_formfactors  = gauss(fit_freqs, high_sig_f)
    new_formfactors = np.append(sorted_formfactors, fit_formfactors)
    new_noise = np.append(sorted_formfactor_noise, np.zeros_like(fit_formfactors))
    new_freqs = np.append(sorted_frequencies, fit_freqs)
    return new_freqs, new_formfactors , new_noise

def smooth_formfactors(formfactors, window_size = 9):
    if window_size >4:
        polyorder = 3
    elif window_size ==3:
        polyorder = 2
    else:
        polyorder = 0
    new_formfactors= savgol_filter(formfactors, window_size,polyorder)
    return new_formfactors
    
def make_linear_spacing(new_frequencies, sorted_frequencies, sorted_formfactors, sorted_formfactors_noise):
    #Interpolation
    f = interpolate.interp1d(sorted_frequencies,sorted_formfactors, bounds_error = True)
    new_formfactors = f(new_frequencies)
    g = interpolate.interp1d(sorted_frequencies, sorted_formfactors_noise, bounds_error = True)
    new_noise = g(new_frequencies)
    return new_formfactors, new_noise

    
"""
Models to fit the formfactors to. 
"""

def rechteck(x, sigma):
    data = super_gauss(x,1,50,sigma)
    return data

def unsym_gauss(x, sigma1,sigma2):
    data = np.zeros(np.size(x))
    data[x>=0]  = gauss(x[x>=0],sigma1)
    data[x<0] = gauss(x [x<0], sigma2)
    return data

def cauchy(x, sigma):
    return 1/ ( 1+(x/sigma)**2)

def unsym_cauchy(x, sigma1,sigma2):
    data = np.zeros(np.size(x))
    data[x>=0]  = cauchy(x [x>=0], sigma1)
    data[x<0] = cauchy(x [x<0], sigma2)
    return data

def saw_tooth(x, sigma):
    data  = 1-0.5/sigma *x 
    data[x<0] = 0
    data[data<0] = 0
    return (data)

def triangular(x,sigma1,sigma2):#
    data = np.zeros(np.size(x))
    data[x>=0]  = 1-0.5/sigma1 *x [x>=0]
    data[x<0] = 1+ 0.5/sigma2 *x [x<0]
    data[data<0] = 0
    return (data)

def exp_step(x, sigma_t):
    expo= np.exp(-x/sigma_t)
    expo[x<0] = 0
    return expo

#List of the models with proper start paras
guess_width= 100e-15
modellist = [[gauss,  [guess_width]],
             [rechteck, [guess_width]],
             [unsym_gauss, [guess_width, guess_width]],
             [unsym_cauchy, [guess_width, guess_width]],
             [saw_tooth, [guess_width]],
             [triangular, [guess_width, guess_width]],   
             [exp_step, [guess_width]],             
]
             

"""
To find good model start
"""

def ff_from_model(model, freqs, modelparas):
    """
    Returns the normalised imaginary formfactor of the model
    """
    #print( model )
    global wx
    zeit = np.fft.fftfreq(2*np.size(freqs)-1, d = freqs[1]-freqs[0])
    zeit = np.sort(zeit)
    if np.size(modelparas) ==2 :
        zeitdom = model(zeit, modelparas[0], modelparas[1])
    else:
        zeitdom= model(zeit, modelparas)
    #normalize to densitiy 1
    zeitdom =zeitdom / np.trapz(zeitdom, x= zeit)
    #print (modelparas)
    #zeitdom= model(zeit, *[modelparas])
    #ff_model = np.fft.fft(zeitdom)
    wx, ff_model = nils_fft(zeit, zeitdom)
    ff_model  = ff_model [:np.size(freqs)] * np.sqrt(2*np.pi)
    #ff_abs_model = np.abs(ff_model)
    #ff_model = ff_model / ff_abs_model[0]
    return ff_model

def fit_model(modeldata, freqs, formfactors, highest_freq):
    """
    Function to fit a model such that the formfactor modulus fits best to the 
    data up to highest freq
    """
    #global model, model_guess
    model = modeldata[0]
    model_guess = modeldata[1]
    if np.size(model_guess) ==2:
        def this_function_to_fit(freqs, sigma1, sigma2):
            sigmas = [sigma1, sigma2]
            this_ff = ff_from_model(model, freqs, sigmas)
            return np.abs(this_ff)
    else:
        def this_function_to_fit(freqs, sigma):
            this_ff = ff_from_model(model, freqs, sigma)
            return np.abs(this_ff)        
        
    formfactors_to_fit = formfactors[freqs<highest_freq]
    freqs_to_fit = freqs[freqs<highest_freq]
    
    popt,pcov = curve_fit(this_function_to_fit, freqs_to_fit, formfactors_to_fit ,p0 = model_guess)
    
    #Get goodness of fit
    #if np.size(popt) ==2:
    #    resulting_ff_abs = np.abs(ff_from_model(model, freqs_to_fit, popt[0], popt[1]))
    #else:
    resulting_ff_abs = np.abs(ff_from_model(model, freqs_to_fit, popt))
    criterium = np.sum((resulting_ff_abs-formfactors_to_fit)**2)
    return criterium, popt

def find_best_modelff(modellist, freqs, formfactors, highest_freq):
    """
    Find the best model from the list of models and returns that formfactor
    """
    n_models = np.shape(modellist)[0]
    gueten = []
    weiten = []
    for n in np.arange(n_models):
        this_res = fit_model(modellist[n], freqs, formfactors, highest_freq)
        gueten = np.append(gueten,this_res[0])
        weiten.append( this_res[1])
    #get the best suited one
    best_model = np.argmin(gueten)
    print('Best found model is ' + modellist[best_model][0].__name__)
    
    #get that models formfactor and return it
    return ff_from_model(modellist[best_model][0], freqs,weiten[best_model])

"""
GERCHBERG-SAXTON RECONSTRUCTION ALGORITHM
"""
def gerchberg_saxton(formfactors, formfactors_noise, start_phase, phase_condition = np.pi/0.01):
    """
    Goes through the iterative phase adjustment until the phase differene is 
    below phase_condition
    """
    j= np.complex(0,1)
    seed_phase = start_phase
    difference = 1e6
    new_formfactor = np.copy(formfactors)
    n = 0    
    nmax = 20
    while n<nmax:
    #while difference > phase_condition:
        prev_phase = np.copy(seed_phase)
        #Nur Formfaktoren austauschen, die nicht im Rauschen liegen
        new_formfactor = np.abs(new_formfactor)
        bools_too_low = new_formfactor<(formfactors - formfactors_noise)
        bools_too_high = new_formfactor>(formfactors + formfactors_noise)
        new_formfactor[bools_too_low]= formfactors[bools_too_low]
        new_formfactor[bools_too_high]= formfactors[bools_too_high]
        
        comp_formfactor = new_formfactor * np.exp(j*seed_phase)
        #erweitern, damit auch negative frequenzen etc. dabei sind. Diese Sortierung ist
        # wichtig, damit ein reales Zeitprofil rauskommt
        comp_formfactor = np.append(comp_formfactor,np.conj(comp_formfactor[1::])[::-1])
        
        #Fourier Transform to get current profile
        this_curr_prof = np.fft.ifft(comp_formfactor)
        this_curr_prof = np.real(this_curr_prof)
        this_curr_prof = np.roll(this_curr_prof, - int(np.argmax(this_curr_prof)-np.size(this_curr_prof)/2))
        #plt.figure('Iteration')
        #plt.plot(this_curr_prof, label = str(n))
        #plt.legend()
        begin_curr_prof = np.copy(this_curr_prof)
        #Error-Reduction-Method: Set to 0 where violating
        #a) negative charges
        #violation = np.any(this_curr_prof<-1e-2)
        this_curr_prof[this_curr_prof<0] = 0
        #b) being larger than 1ps
        #center = first_moment(this_curr_prof, zeit)
        #this_curr_prof[:100] = 0
        #this_curr_prof[-100:] = 0
        #c) zusammenhängend
        # get rid of postoscillations...
        #check low
        x = np.arange(np.size(this_curr_prof)/2)
        low_positions = x[this_curr_prof[:np.size(x)]==0]
        high_positions = x[this_curr_prof[np.size(x)-1:]==0]
        if np.size(low_positions)==0:
            low_index = 0
        else:
            low_index =low_positions[-1]
        if np.size(high_positions)==0:
            high_index = x[-1]
        else:
            high_index =high_positions[0] + x[-1]
        this_curr_prof[:int(low_index)] = 0
        this_curr_prof[int(high_index):] = 0
        #..done

        #plt.plot(this_curr_prof, label=step)
        #FFT to get new phase in time domain
        x  = np.arange(np.size(this_curr_prof))
        """
        Change here
        """
        this_curr_prof = this_curr_prof #/np.trapz(this_curr_prof, x = x)
        """
        """
        #new_formfactor = np.fft.fft(this_curr_prof)
        wx, new_formfactor = nils_fft(x,this_curr_prof)
        new_formfactor = new_formfactor * np.sqrt(2*np.pi)
        new_formfactor = new_formfactor[:int((np.size(new_formfactor)+1)/2)]
        seed_phase = np.angle(new_formfactor)
        difference = np.sum(np.abs(seed_phase-prev_phase)**2)
        #difference = np.sum(np.abs(np.abs(new_formfactor[:20])-np.abs(inter_formfactors)))
        #Wenn ich so aufhöre, stimmt der Formfakor bei hohen Frequenzen nicht mehr überein:
        #if difference <=phase_condition:
        if n == nmax-1:
            clean_curr_prof = np.copy(this_curr_prof) # mit Cleanup
            this_curr_prof = begin_curr_prof # ohne Clean up
            x  = np.arange(np.size(this_curr_prof))
            #outcommented this
            #this_curr_prof = this_curr_prof /np.trapz(this_curr_prof, x = x)
            #new_formfactor = np.fft.fft(this_curr_prof)
            #wx, new_formfactor = nils_fft(x,this_curr_prof)
            #new_formfactor = new_formfactor * np.sqrt(2*np.pi)
            #new_formfactor = new_formfactor[:int((np.size(new_formfactor)+1)/2)]
        n= n +1
            
        
    #this_curr_prof = np.roll(this_curr_prof, - int(np.argmax(this_curr_prof)-np.size(this_curr_prof)/2))
    #center fine
    #center = first_moment(this_curr_prof, np.arange(np.size(this_curr_prof)))
    center = first_moment(clean_curr_prof, np.arange(np.size(clean_curr_prof)))
    final_curr_prof = np.roll(this_curr_prof, -int(center-np.size(this_curr_prof)/2))
    clean_curr_prof = np.roll(clean_curr_prof, -int(center-np.size(clean_curr_prof)/2))
    return  clean_curr_prof#, final_curr_prof # # mit und ohne clean up

def kramers_kronig_phase(frequencies, mess_formfactors):
    """
    calculate the kramers kronig phase. Input frequencies and formfactors must be sorted
    """
    #Ln works only for positiv Formfactors
    these_freqs = frequencies[mess_formfactors>0]
    these_ff = mess_formfactors[mess_formfactors>0]
    indices = np.arange(np.size(frequencies))[mess_formfactors>0]
    kk_phase = np.zeros_like(frequencies)
    for i in np.arange(np.size(indices)):
        #avouid 0 in nenner
        ff_here = np.delete(these_ff,i)
        freqs_here =np.delete(these_freqs,i)
        to_integrate = (np.log(ff_here )- np.log(these_ff [i])) /(these_freqs[i]**2 - freqs_here**2)
        kk_phase[indices[i]] =  2 * these_freqs[i] / np.pi * np.trapz(to_integrate, x = freqs_here)
    
    return kk_phase
        

def model_start(frequencies, model_ff, mess_formfactors, mess_formfactors_noise):
    """
    Starts the Gerchberg Saxton algorithm with a complex ff and gives the final time profil
    """
    
    zeit = np.fft.fftfreq(2*np.size(frequencies)-1, d = frequencies[1]-frequencies[0])
    zeit = np.sort(zeit)
    #tmax = 1/(frequencies[0]-frequencies[1])
    #zeit = tmax*np.linspace(-1/2,1/2,np.size(frequencies))
    start_phase = np.angle(model_ff)
    time_domain = gerchberg_saxton(mess_formfactors, mess_formfactors_noise, start_phase)
    return zeit, time_domain

def norm_current(zeit, profil, charge):
    """
    Normalizes the longitudinal profile to the actual charge
    """
    return profil/np.trapz(profil, x = zeit) *charge


def cleanup_formfactor (frequencies, formfactors, formfactors_noise, formfactors_det_limit, 
                        channels_to_remove = [104,135], wanted_time_res = 2e-15, wanted_time_frame = 2e-12, 
                     high_inter_last = 1, model_last_index =  -1, smooth_window = 9):
    """
    These function cleans up the form factor and interpolates it in frequency
    """
    frequencies, formfactors, formfactors_noise, formfactors_det_limit = remove_certain_channels(channels_to_remove, [frequencies, formfactors,formfactors_noise, formfactors_det_limit])
    sort_freq, sorted_stuff = sort_data(frequencies, [formfactors,  formfactors_noise,formfactors_det_limit])
    sign_frequencies, sign_formfactors, sign_formfactors_noise = remove_loners(sort_freq, sorted_stuff[0], sorted_stuff[1], sorted_stuff[2])
    #New FrequencyGrid
    freq_d = 1/wanted_time_frame
    freq_max = 1/wanted_time_res
    n_freqs = int(freq_max/freq_d)
    #Frequencies must contain 0!
    new_freqs = np.linspace(0,freq_max, n_freqs)
    #Extrapolations
    extra_low_freqs, extra_low_ff, extra_low_ff_noise = extrapolation_low(new_freqs,sign_frequencies, sign_formfactors,sign_formfactors_noise)
    extra_high_freqs, extra_high_ff, extra_high_ff_noise = extrapolation_high(new_freqs,extra_low_freqs, extra_low_ff, extra_low_ff_noise, index_to_cut = high_inter_last)
    #Smoothing & Interpolation
    smooth_ff = smooth_formfactors(extra_high_ff, window_size=smooth_window)
    final_ff, final_ff_noise = make_linear_spacing(new_freqs, extra_high_freqs, smooth_ff, extra_high_ff_noise)
    return new_freqs, final_ff, final_ff_noise

def master_recon(frequencies, formfactors, formfactors_noise, formfactors_det_limit, charge, method = 'KKstart',
                 channels_to_remove = [104,135], wanted_time_res = 2e-15, wanted_time_frame = 2e-12, 
                 high_inter_last = 1, model_last_index =  -1, smooth_window = 9, phase_noise_start =None,show_plots = True):
    """
    Master Function to reconstruct the current profile from a given formfactor
    measurement. Returns time and current profile
    """
    
    frequencies, formfactors, formfactors_noise, formfactors_det_limit = remove_certain_channels(channels_to_remove, [frequencies, formfactors,formfactors_noise, formfactors_det_limit])
    sort_freq, sorted_stuff = sort_data(frequencies, [formfactors,  formfactors_noise,formfactors_det_limit])
    sign_frequencies, sign_formfactors, sign_formfactors_noise = remove_loners(sort_freq, sorted_stuff[0], sorted_stuff[1], sorted_stuff[2])
    #New FrequencyGrid
    freq_d = 1/wanted_time_frame
    freq_max = 1/wanted_time_res
    n_freqs = int(freq_max/freq_d)
    #Frequencies must contain 0!
    new_freqs = np.linspace(0,freq_max, n_freqs)
    #Extrapolations
    extra_low_freqs, extra_low_ff, extra_low_ff_noise = extrapolation_low(new_freqs,sign_frequencies, sign_formfactors,sign_formfactors_noise)
    extra_high_freqs, extra_high_ff, extra_high_ff_noise = extrapolation_high(new_freqs,extra_low_freqs, extra_low_ff, extra_low_ff_noise, index_to_cut = high_inter_last)
    #Smoothing & Interpolation
    smooth_ff = smooth_formfactors(extra_high_ff, window_size=smooth_window)
    final_ff, final_ff_noise = make_linear_spacing(new_freqs, extra_high_freqs, smooth_ff, extra_high_ff_noise)
    if method == 'modell':
        #Modelfit
        start_ff = find_best_modelff(modellist, new_freqs, final_ff, sign_frequencies[model_last_index])
        #return start_ff
    elif method == 'KKstart':
        #print('Drin')
        kk_phase = kramers_kronig_phase(new_freqs, final_ff)
        start_ff = final_ff * np.exp(kk_phase*complex(0,1))
        #return start_ff
    else: 
        print('Method not defined')
    if phase_noise_start !=None:
        #alle start phasen ueber 20THz random verteilen
        crit_freq = phase_noise_start
        n_phases = np.size(new_freqs[new_freqs>crit_freq])
        rand_phases = np.random.rand(n_phases)*2*np.pi
        start_ff[new_freqs>crit_freq] = np.abs(start_ff[new_freqs>crit_freq]) * np.exp(complex(0,1)*rand_phases)
        
    #Reconstruction
    recon_time, recon_prof = model_start(new_freqs, start_ff, final_ff, final_ff_noise)
    #Normalization to current
    current = norm_current(recon_time, recon_prof, charge)
    
    #get formfactor of reconstruction
    #wx, recon_ff = nils_fft(recon_time,current/charge)
    wx, recon_ff = nils_fft(np.arange(np.size(recon_time)),recon_prof)
    recon_ff = recon_ff * np.sqrt(2*np.pi)
    recon_ff = recon_ff[:int((np.size(recon_ff)+1)/2)]
    recon_ff = np.abs(recon_ff)
    
    #RMS Time Value
    t_rms = np.sqrt(np.abs(second_moment(current, recon_time)))
    if show_plots == True:
        fig,axes = plt.subplots(nrows =  2, ncols = 1)
        #Raw Data
        axes[0].errorbar(sort_freq*1e-12, sorted_stuff[0], yerr = sorted_stuff[1], fmt = 'o', label = 'Raw Data', color = 'b')
        axes[0].fill_between(sort_freq*1e-12, 1e-4, sorted_stuff[2], label = 'Det Limit', color = 'gray', alpha = 0.3)
        #Before Recon
        axes[0].errorbar(new_freqs*1e-12, final_ff,yerr = final_ff_noise, fmt ='-', label = 'Input to Recon.', color = 'orange')
        #Modell
        if method == 'modell':
            axes[0].plot(new_freqs*1e-12, np.abs(start_ff), label = 'Model', color = 'g')
        #Results of Recon
        axes[0].plot(new_freqs*1e-12, recon_ff, '--', label = 'Result of Recon.', color = 'r')
        #To look nice
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].set_xlim([0.5,100])
        axes[0].set_xlabel('f (THz)')
        axes[0].set_ylabel('|F|')
        axes[0].set_ylim([5e-3,2])
        axes[0].legend(loc = 'lower left')
        
        #Time Domain
        axes[1].plot(recon_time*1e15, current*1e-3, label = r'$\sigma_t = $' + str(int(round(t_rms*1e15)))+ ' fs')
        axes[1].set_xlabel('t (fs)')
        axes[1].set_ylabel('I (kA)')
        axes[1].set_xlim([-10*t_rms*1e15,10*t_rms*1e15])
        #axes[1].set_xlim([-200,200])
        axes[1].legend()
        fig.tight_layout()

    return recon_time, current, t_rms


