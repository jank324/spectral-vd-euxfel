import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from simulate_spectrometer_signal import *
from reconstruction_module_final import *


plt.close('all')
s = np.linspace(-100, 100, 240) # in [um]

charge = 250e-12


def profile(s, params, charge):
    a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3 = params
    p = np.zeros(len(s))
    p += a1 * np.exp(-(s - mu1) ** 2 / 2 / sigma1 ** 2)

    p += a2 * np.exp(-(s - mu2) ** 2 / 2 / sigma2 ** 2)
    p += a3 * np.exp(-(s - mu3) ** 2 / 2 / sigma3 ** 2)

    p = gaussian_filter1d(p, 2)
    current = p / np.trapz(p, x=s / 3e8) * charge / 1e-6
    return current


def random_current_profiles(n_profiles=10, charge=250e-12):

    params = np.zeros((n_profiles, 9))
    profiles = []

    for i in range(n_profiles):
        sigma1 = np.random.random() * 5
        mu1 = np.random.randint(low=-30, high=30 + 1)
        a1 = np.random.random()
        sigma2 = np.random.random() * 5
        mu2 = np.random.randint(low=-30, high=30 + 1)
        a2 = np.random.random()

        sigma3 = np.random.randint(low=7, high=30+1)
        mu3 = np.random.randint(low=-20, high=20+1)
        a3 = np.random.random()/2 + 0.5
        param = [a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3]
        current = profile(s, param, charge)
        profiles.append(current)
        params[i, :] = param
    return profiles, params


if __name__ == "__main__":
    np.random.seed(10)

    NPROFILES = 1000#50000

    profiles, params = random_current_profiles(n_profiles=NPROFILES, charge=250e-12)

    plt.figure()
    plt.title('Electron  Bunch')
    plt.xlabel('s (µm)')
    plt.ylabel('I (kA)')
    for i, p in enumerate(profiles[:5]):
        plt.plot(s, profile(s, params[i], charge)*1e-3 )


    plt.figure()
    plt.title('Spectrometer Formfactor')
    plt.xlabel('f (THz)')
    plt.ylabel('|F|')
    all_ff=[]
    for p in profiles[:5]:
        frequencies, formfactors,det_lim = get_crisp_signal(s*1e-6, p, n_shots=10, which_set='both')
        #formfactors[formfactors < 0.01] = 0
        plt.loglog(frequencies*1e-12, formfactors)
        all_ff.append(formfactors)
    plt.fill_between(frequencies*1e-12, y1 = 1e-3,y2 =det_lim)        

    plt.show()

    recons = []
    formfactor_noise = np.zeros(np.size(all_ff[0]))
    for i in [0,1,2,3,4]:#,1999,2999,3999,4999]:
        t,cur,rms = master_recon(frequencies, all_ff[i], formfactor_noise, det_lim,
                  charge, method = 'KKstart',channels_to_remove = [], 
                  wanted_time_res = 2e-15, wanted_time_frame = 2e-12, high_inter_last = 1, 
                  model_last_index =  -1, smooth_window = 5,extra_low_ff_value = 0.66, show_plots = True)
        recons.append(cur)
    
    #compare with inital
    plt.figure()
    for i in np.arange(5):
        #center initial profiles
        center = first_moment(profiles[i],s)
        p1= plt.plot(s-center, profiles[i]*1e-3,label = str(i))
        plt.plot(-3e8*t*1e6, recons[i]*1e-3,'--', color = p1[0].get_color())
    plt.legend()
    plt.xlabel(u's (µm)')
    plt.ylabel('I (kA)')
    plt.xlim([-100,100])
    
    #check how long bunches may become
    elongate_factors = [1,2,5,8,10]
    longer_recons = []
    longer_rms = []
    plt.figure('Longer FF')
    for longer in elongate_factors:
        this_s = s*longer
        frequencies, formfactor,det_lim = get_crisp_signal(this_s*1e-6, profiles[3]/longer, n_shots=50, which_set='both')
        t,cur,rms = master_recon(frequencies, formfactor, formfactor_noise, det_lim,
          charge, method = 'KKstart',channels_to_remove = [], 
          wanted_time_res = 2e-15, wanted_time_frame = 3e-12, high_inter_last = 1, 
          model_last_index =  -1, smooth_window = 5,extra_low_ff_value = 0.5, show_plots = True)
        longer_recons.append(cur)
        longer_rms.append(rms)
        plt.loglog(frequencies*1e-12, formfactor)
    plt.figure('Longer')
    i=0
    for longer in elongate_factors:
        p2 = plt.plot(s*longer, profiles[3]/longer)
        plt.plot(3e8*t*1e6, longer_recons[i],'--', color = p2[0].get_color(), label  =str (longer_rms[i]*1e15)+ ' fs')
        i=i+1
    plt.legend()