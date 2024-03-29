#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:40:06 2019
Function to simulate the measured formfactor of the spectrometer for input of
s (m) and I (amps)

@author: lockmann
"""
import numpy as np
import matplotlib.pylab as plt

"""
simulate spectrometer
"""

def get_crisp_signal(s, current, n_shots = 1, which_set = 'high'):
    """Simulates the signal of the spectrometer of a single shot including the 
    noise of the spectrometer. 
    
    Parameters
    ----------
    s :         numpy array of floats
                longitudinal bunch coordinate.
    current :   numpy array of floats with same length as s
                current in ampere
    n_shots:    int, optional
                number of shots to average from. Reduces spectrometer noise. 
    which_set:  str, optional
                Specifies the grating set of the spectrometer to use. Keep in 
                mind that experimentally you can only get one at a time. 
                Keywords are: 'high', 'low' and 'both'
    
    Returns
    ------
    results :   float (array)
                resulting electric field at distance r
    """
    #center s so that 0 is in the middle
    s_centered = s - np.sum(s*current)/np.sum(current)
    # get charge
    t_in = s_centered/3e8
    charge = np.trapz(current, x= t_in)
    #linear time spacing
    t_spec = np.arange(-2,2,0.001)*1e-12
    #needs to be sorted
    current = current[np.argsort(t_in)]
    t_in = np.sort(t_in)
    cur_spec = np.interp(t_spec, t_in, current, left=0, right=0)
    #normalize to one
    cur_spec = cur_spec/np.trapz(cur_spec,x=t_spec)
    
    #plt.figure('Normalized and intrapolated')
    #plt.plot(t_spec*1e15,cur_spec*1e-3)
    
    # fft
    comp_ff = np.fft.fft(cur_spec)
    scale_factor =  (t_spec[1]- t_spec[0])
    comp_ff = comp_ff*scale_factor
    freqs = np.fft.fftfreq(np.size(cur_spec),t_spec[1]-t_spec[0])
    comp_ff = comp_ff[freqs>=0]
    freqs = freqs [freqs>=0]
    
    #plt.figure()
    #plt.plot(freqs*1e-12,np.abs(comp_ff))
    #plt.xscale('log')
    #plt.yscale('log')
    
    #interpolate to spectrometer range
    freqs_spectrometer = np.array([6.84283010e+11, 6.86342276e+11, 6.89291238e+11, 6.93231268e+11,
           6.97867818e+11, 7.03183896e+11, 7.09148197e+11, 7.15734438e+11,
           7.23107481e+11, 7.31240520e+11, 7.40197234e+11, 7.50015830e+11,
           7.60740352e+11, 7.72411055e+11, 7.85016183e+11, 7.98657943e+11,
           8.13405379e+11, 8.29377459e+11, 8.46469777e+11, 8.65200287e+11,
           8.85430610e+11, 9.07299763e+11, 9.31292658e+11, 9.57343678e+11,
           9.85437362e+11, 1.01603519e+12, 1.04944212e+12, 1.08594355e+12,
           1.12511418e+12, 1.16825162e+12, 1.25452235e+12, 1.25941010e+12,
           1.26511563e+12, 1.27198544e+12, 1.28040955e+12, 1.29006363e+12,
           1.30097204e+12, 1.31295400e+12, 1.32651050e+12, 1.34139271e+12,
           1.35760346e+12, 1.37547358e+12, 1.39503008e+12, 1.41656334e+12,
           1.43974321e+12, 1.46457178e+12, 1.49167731e+12, 1.52101420e+12,
           1.55222847e+12, 1.58655331e+12, 1.62359096e+12, 1.66369362e+12,
           1.70718601e+12, 1.75508152e+12, 1.80652971e+12, 1.86252358e+12,
           1.92362000e+12, 1.99001009e+12, 2.06242973e+12, 2.14196419e+12,
           2.28119024e+12, 2.28903842e+12, 2.29950939e+12, 2.31261258e+12,
           2.32772918e+12, 2.34515972e+12, 2.36498904e+12, 2.38681143e+12,
           2.41113054e+12, 2.43827617e+12, 2.46831212e+12, 2.50106736e+12,
           2.53639935e+12, 2.57499338e+12, 2.61694409e+12, 2.66265777e+12,
           2.71189970e+12, 2.76495993e+12, 2.82181770e+12, 2.88430880e+12,
           2.95157763e+12, 3.02570939e+12, 3.10515027e+12, 3.19080363e+12,
           3.28411150e+12, 3.38505628e+12, 3.49557917e+12, 3.61569473e+12,
           3.74494818e+12, 3.87221178e+12, 3.87605079e+12, 3.89182771e+12,
           3.91112792e+12, 3.93302151e+12, 3.95880137e+12, 3.98757241e+12,
           4.02090481e+12, 4.05868059e+12, 4.09887219e+12, 4.14427297e+12,
           4.19590008e+12, 4.25159913e+12, 4.31184945e+12, 4.37798056e+12,
           4.45017789e+12, 4.52767991e+12, 4.61054903e+12, 4.70100308e+12,
           4.79815306e+12, 4.90363011e+12, 5.01847576e+12, 5.14289228e+12,
           5.27853835e+12, 5.42497233e+12, 5.58394521e+12, 5.75566213e+12,
           5.94270061e+12, 6.14675659e+12, 6.36808841e+12, 6.60398418e+12,
           6.84239080e+12, 6.86645607e+12, 6.89844358e+12, 6.93795435e+12,
           6.98381615e+12, 7.03665301e+12, 7.09581595e+12, 7.16182658e+12,
           7.23558808e+12, 7.31656935e+12, 7.40588391e+12, 7.50366620e+12,
           7.61017924e+12, 7.72725265e+12, 7.85313450e+12, 7.98952729e+12,
           8.13636493e+12, 8.29743559e+12, 8.46741872e+12, 8.65427554e+12,
           8.85635877e+12, 9.07462058e+12, 9.31463114e+12, 9.57583282e+12,
           9.85392517e+12, 1.01581001e+13, 1.04872716e+13, 1.08470040e+13,
           1.12261054e+13, 1.13980414e+13, 1.14448608e+13, 1.15007118e+13,
           1.15668861e+13, 1.16436805e+13, 1.16745582e+13, 1.17301474e+13,
           1.18286343e+13, 1.19375252e+13, 1.20584689e+13, 1.21926665e+13,
           1.23420542e+13, 1.25037214e+13, 1.26827481e+13, 1.28758007e+13,
           1.30836435e+13, 1.33098200e+13, 1.35576821e+13, 1.38245741e+13,
           1.41103234e+13, 1.44217346e+13, 1.47624195e+13, 1.51316322e+13,
           1.55258557e+13, 1.59552050e+13, 1.64196917e+13, 1.69285923e+13,
           1.74782780e+13, 1.80802622e+13, 1.87316513e+13, 1.94415752e+13,
           2.05304449e+13, 2.06041596e+13, 2.06980683e+13, 2.08140040e+13,
           2.09510109e+13, 2.11093505e+13, 2.12864874e+13, 2.14804199e+13,
           2.17008944e+13, 2.19489705e+13, 2.22216055e+13, 2.25161992e+13,
           2.28301018e+13, 2.31801788e+13, 2.35588868e+13, 2.39691136e+13,
           2.44111492e+13, 2.48897567e+13, 2.54005500e+13, 2.59618741e+13,
           2.65693466e+13, 2.72227653e+13, 2.79406515e+13, 2.87222610e+13,
           2.95520370e+13, 3.04799241e+13, 3.14692524e+13, 3.25463980e+13,
           3.36766529e+13, 3.41763020e+13, 3.43153008e+13, 3.44839431e+13,
           3.46803032e+13, 3.49137251e+13, 3.49825754e+13, 3.51760204e+13,
           3.54682908e+13, 3.57958567e+13, 3.61610300e+13, 3.65661507e+13,
           3.70094847e+13, 3.74942425e+13, 3.80275217e+13, 3.86055890e+13,
           3.92357545e+13, 3.99218111e+13, 4.06583840e+13, 4.14533934e+13,
           4.23052569e+13, 4.32362878e+13, 4.42368558e+13, 4.53735986e+13,
           4.65444168e+13, 4.78371060e+13, 4.92264177e+13, 5.07656062e+13,
           5.24232320e+13, 5.41042876e+13, 5.61221152e+13, 5.82673400e+13])
    
    ff_spectrometer = np.interp(freqs_spectrometer, freqs, np.abs(comp_ff))
    
    #plt.figure('Spectrometer Signal')
    #plt.plot(freqs_spectrometer*1e-12, ff_spectrometer, 'o')
    #plt.xscale('log')
    #plt.yscale('log')
    
    #add spectrometer noise
    #response in V/nC²
    spec_response = np.array([2.27139720e-02, 2.63574891e-02, 1.91519673e-02, 2.68880698e-02,
           2.96823444e-02, 2.50777417e-02, 5.14215877e-02, 3.65048653e-02,
           5.01399530e-02, 4.92891672e-02, 6.28404705e-02, 5.90190272e-02,
           7.09957402e-02, 9.88634708e-02, 1.14630994e-01, 1.23265766e-02,
           1.23461331e-01, 1.36320335e-01, 1.25770119e-01, 1.32215392e-01,
           1.36440279e-01, 1.34883965e-01, 1.31848572e-01, 1.64531877e-01,
           1.55915488e-01, 1.92955737e-01, 2.10324614e-01, 2.06284403e-01,
           1.21629041e-01, 1.28159124e-01, 2.82388474e-02, 3.24896486e-02,
           6.32553505e-02, 7.70183540e-02, 8.65472943e-02, 1.15282182e-01,
           1.20753130e-01, 1.58432901e-01, 1.75611721e-01, 1.91465810e-01,
           2.57557223e-01, 2.96939075e-01, 3.14254548e-01, 3.66881036e-01,
           3.91964519e-01, 3.94565181e-01, 3.35777982e-01, 4.34468414e-01,
           4.34228563e-01, 4.55406870e-01, 6.28887904e-01, 6.28342309e-01,
           7.46633419e-01, 7.66892682e-01, 7.07245311e-01, 6.48177114e-01,
           4.44611893e-01, 5.51623941e-01, 7.26412461e-01, 6.13676906e-01,
           6.79542586e-02, 8.16807490e-02, 1.32899393e-01, 1.91507017e-01,
           1.90696482e-01, 2.00147063e-01, 2.73266787e-01, 2.51203348e-01,
           2.91641304e-01, 3.02340215e-01, 2.90033553e-01, 3.56812708e-01,
           3.80332636e-01, 4.71988270e-01, 5.71053105e-01, 5.76061366e-01,
           7.42794124e-01, 8.28076940e-01, 8.61271616e-01, 9.34415109e-01,
           9.88661829e-01, 1.29302949e+00, 1.68370088e+00, 1.60985763e+00,
           1.66388597e+00, 2.18810117e+00, 1.77063413e+00, 2.22234386e+00,
           1.79544513e+00, 6.49211052e-01, 1.12905733e-01, 1.35219749e-01,
           1.77552881e-01, 2.33864266e-01, 2.76136407e-01, 3.08371345e-01,
           4.22577865e-01, 4.50832291e-01, 4.07343365e-01, 4.54518985e-01,
           4.74195250e-01, 5.14916496e-01, 5.82809637e-01, 6.85511764e-01,
           7.85862983e-01, 9.18192559e-01, 1.08807514e+00, 1.32820958e+00,
           1.63915706e+00, 1.88506672e+00, 2.11140637e+00, 2.56819078e+00,
           3.47029067e+00, 4.43909591e+00, 4.59933654e+00, 5.20503438e+00,
           4.63311775e+00, 4.98549453e+00, 6.05139362e+00, 2.00813681e+00,
           3.17399571e-01, 4.80405496e-01, 6.33070146e-01, 7.01504333e-01,
           8.78360578e-01, 1.04252778e+00, 1.33420246e+00, 1.44414774e+00,
           1.60301216e+00, 1.95838326e+00, 1.97299122e+00, 2.01505983e+00,
           2.48201305e+00, 3.35207178e+00, 4.50221954e+00, 1.02154423e-01,
           5.83456121e+00, 6.21937385e+00, 6.72302208e+00, 6.93572009e+00,
           7.72672143e+00, 9.53759783e+00, 9.70607794e+00, 9.88128478e+00,
           1.02574353e+01, 1.17366859e+01, 9.75250128e+00, 8.49536317e+00,
           5.87634647e+00, 6.90542536e-01, 8.84953809e-01, 1.10143838e+00,
           1.32548040e+00, 1.39920693e+00, 2.64741690e-01, 1.49174256e+00,
           1.42851071e+00, 1.61623263e+00, 2.14723132e+00, 2.49236663e+00,
           2.41236131e+00, 2.52993530e+00, 1.95052864e+00, 2.08077797e+00,
           2.46425871e+00, 1.86427999e+00, 1.29883560e+00, 1.61806660e+00,
           2.69349944e+00, 5.26651344e+00, 8.60076710e+00, 1.23955411e+01,
           1.55537653e+01, 1.57520786e+01, 1.55318570e+01, 1.63010277e+01,
           1.95997146e+01, 1.89657798e+01, 1.98173958e+01, 1.28444056e+01,
           1.05663513e+00, 1.73837417e+00, 2.38571066e+00, 3.12509262e+00,
           5.16370343e+00, 6.27279615e+00, 4.55271819e+00, 2.30736012e+00,
           1.00261293e+00, 1.28871366e+00, 6.82622230e+00, 1.52316196e+01,
           1.30645368e+01, 1.56890352e+01, 1.39606414e+01, 1.52718010e+01,
           7.30423998e+00, 1.86593830e+01, 2.01944545e+01, 2.18768013e+01,
           2.09667009e+01, 2.15890054e+01, 2.39164749e+01, 2.48425545e+01,
           2.30255779e+01, 1.71761298e+01, 2.33330662e+01, 2.32725960e+01,
           2.10340416e+01, 5.24993968e-01, 1.63131206e+00, 3.44781069e+00,
           2.29974095e+00, 4.99462286e+00, 2.32302429e+00, 4.96744300e+00,
           2.68405786e+00, 8.62683761e+00, 1.15968873e+01, 1.43637056e+01,
           1.16110817e+01, 9.31426757e+00, 1.07818775e+01, 1.02543508e+01,
           7.20074507e+00, 1.01373496e+01, 1.66319551e+01, 5.87900394e+00,
           1.78322017e+01, 6.29817365e+00, 9.87433101e-01, 8.83113064e+00,
           6.72258056e+00, 7.85860449e+00, 9.53416250e+00, 2.59286607e+01,
           2.66550560e+01, 2.05970702e+01, 2.45164563e+00, 1.38458228e+00])
    
    #formfactor to adc-signal
    adc_sig = ff_spectrometer**2 * (charge*1e9)**2 * spec_response # in V
    #plt.figure('ADC Signal')
    #plt.plot(adc_sig)
    #add_noise
    elec_noise = 1.2e-3 #V
    elec_noise = elec_noise/np.sqrt(n_shots)
    adc_noise = np.random.randn(np.size(ff_spectrometer))*elec_noise
    adc_total = adc_sig + adc_noise
    
    #And back to formfactor
    final_ff = 1/(charge*1e9)*np.sqrt(np.abs(adc_total)/spec_response) *np.sign(adc_total)
    #noise on form factor
    ff_noise  = 0.5 * final_ff * elec_noise * np.sqrt(n_shots) / adc_total #fehlerfortpflanzung
    det_lim  =  1/(charge*1e9)*np.sqrt(np.abs(elec_noise)/spec_response) # noise floor
    #plt.figure()
    #plt.plot(freqs_spectrometer*1e-12, spec_response, 'o')
    #plt.xscale('log')
    #plt.yscale('log')
    
    #plt.figure('Spectrometer Signal')
    #plt.plot(freqs_spectrometer*1e-12, final_ff, 'o')
    #plt.xscale('log')
    #plt.yscale('log')
    if which_set =='low':    
        freqs_to_return = freqs_spectrometer[:120]
        ff_to_return = final_ff[:120]
        det_lim = det_lim[:120]
        ff_noise = ff_noise[:120]        
    elif which_set == 'high':
        freqs_to_return = freqs_spectrometer[120:]
        ff_to_return = final_ff[120:]
        det_lim = det_lim[120:]
        ff_noise = ff_noise[120:]
    elif which_set == 'both':
        freqs_to_return = freqs_spectrometer
        ff_to_return = final_ff  
        det_lim = det_lim
        ff_noise = ff_noise
    else:
        print('Keyword not known!')
        
    return np.array([freqs_to_return, ff_to_return, ff_noise, det_lim])


