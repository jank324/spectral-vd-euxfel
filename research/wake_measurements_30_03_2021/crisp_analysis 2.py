#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 12:37:49 2020

@author: xfeloper
"""
import numpy as np

import matplotlib.pyplot as plt
import time
import pickle 
import matplotlib
from reconstruction_module import *
from ocelot.utils.wake_estimator import WakeLoss, BeamShape
from ocelot.common.globals import *

font = {#'family' : 'normal',
    #'weight' : 'bold',
    'size'   : 16}

matplotlib.rc('font', **font)

def gauss_fit(x, y, A=1, q=250e-12):
    def gauss(x, *p):
        A, mu, sigma = p
        return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))

    sigma = q * speed_of_light / (A * np.sqrt(np.pi * 2))
    p0 = [A, 0., sigma]

    coeff, var_matrix = curve_fit(gauss, x, y, p0=p0)

    y_gauss = gauss(x, *coeff)
    return y_gauss

crisp_ch = 'XFEL.SDIAG/THZ_SPECTROMETER.FORMFACTOR/CRD.1934.TL/FORMFACTOR.XY'

BCM_B2_ch = "XFEL.FEEDBACK/FT1.LONGITUDINAL/MONITOR12/MEAN"
BCM_B2_2_ch = "XFEL.FEEDBACK/FT1.LONGITUDINAL/MONITOR13/MEAN"
BCM1_B1 = "XFEL.FEEDBACK/FT1.LONGITUDINAL/MONITOR8/MEAN"
L2_chirp_ch = "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L2/SUMVOLTAGE.CHIRP.SP.1"

CL_energy_ch = "XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/CL/ENERGY.ALL"
TLD_energy_ch = "XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/TLD/ENERGY.ALL"
T4_energy_ch = "XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/T4/ENERGY.ALL"
T4D_energy_ch = "XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/T4D/ENERGY.ALL"



SASE3_Eph_ch = "XFEL.FEL/WAVELENGTHCONTROL.SA3/XFEL.SA3/E_PHOTON"


SET_ch = "XFEL.FEL/WAVELENGTHCONTROL.SA3/XFEL.SA3/CMD"
ready_ch = "XFEL.FEL/WAVELENGTHCONTROL.SA3/XFEL.SA3/"
cell2_K = "XFEL.FEL/UNDULATOR.SASE3/U68.2809.SA3/K.SET"


# OK
filename1 = "20210330-19_51_10_stage2_LH_OFF_fix.p"
filename2 = "20210330-20_22_53_stage1_LH_OFF.p"
# Next day
#filename1 = "20210221-03_24_50_CRISP_short_fb_LH_OFF.p"
#filename2 = "20210221-02_55_26_CRISP_long_fb_LH_OFF.p"
# LH 4800
filename1 = "20210330-20_05_57_stage2_LH_4000.p"
filename2 = "20210330-20_38_00_stage1_LH_4000.p"
# LH = 1800
#filename1 = "20210220-04_21_50_CRISP_short_fb_LH_1800.p"
#filename2 = "20210220-04_36_08_CRISP_long_fb_LH_1800.p"


DATA1 = pickle.load(open(filename1, "rb"))
#DATA1 = []
#chirp = []
#for d in DATA1_tmp:
#    ch = d["chirp"]
#    print(ch)
#    if ch in chirp:
#        continue
#    chirp.append(ch)
#    DATA1.append(d)
#pickle.dump(DATA1, open("20210330-19_51_10_stage2_LH_OFF_fix.p", "wb"))

DATA2 = pickle.load(open(filename2, "rb"))
#print(DATA1["chirp"])
#exit()
charge = 0.25 # nC

plt.figure(1)

CRISP_short = []
#f, (ax1, ax2) = plt.subplots(2, 1)
f, ax1 = plt.subplots(1,1)
f, ax2 = plt.subplots(1,1)
#plt.figure(101)

CHIRP = []
LOSS = []
data = {"chirp": [], "freq": [], "FF":[]}
i = 0
for data1, data2 in zip(DATA1, DATA2):
    i += 1
    #if i % 2 ==0:
    #    continue
    #print(data1["CRISP.INT"])
    chirp = data1["chirp"]
    print(data1["chirp"], data2["chirp"])
    data["chirp"].append(chirp)
    #if chirp not in [-9, -9.5, -9.25]:
    #    continue
    FF1 = np.array([d[:, 1] for d in data1["CRISP"]])
    FF2 = np.array([d[:, 1] for d in data2["CRISP"]])

    FF1 = FF1.reshape(-1, 240)
    print(FF1.shape)
    ff1_mean = np.mean(FF1, axis=0)
    FF2 = FF2.reshape(-1, 240)
    ff2_mean = np.mean(FF2, axis=0)
    freq = data1["CRISP"][0][:, 0] *1e12
    data["freq"].append(freq)

    #freq = get_ff_spectrometer()
    n_shots = 50
    FF_all = np.append(FF2[:, :120], FF1[:, 120:], axis=1)
    data["FF"].append(FF_all)
    FF = (np.append(ff2_mean[:120], ff1_mean[120:]))
    FF[FF<0] = 0
    ff = np.sqrt(FF)
    ff_noise, det_lim = get_noise_detlim(final_ff=ff, charge=250e-12, n_shots=n_shots)
    #print(ff_noise)
    newfreqs, newff, newffnoise = cleanup_formfactor(freq, ff,
                                                     ff * 0.1, det_lim,
                                                     channels_to_remove=[])
    noise = np.sqrt(np.std(FF_all, axis=0))
    crisp_recon_time, crisp_recon_current, crisp_t_rms = master_recon(freq, ff, noise,
                                                                      det_lim, charge=0.25e-9,
                                                                      channels_to_remove=[], show_plots=False)

    ax1.plot(crisp_recon_time*1e15, crisp_recon_current*1e-3, label="L1 chirp = " + str(chirp))
    ax2.plot(freq*1e-12, FF,  label="L1 chirp = " + str(chirp))
    s = crisp_recon_time*speed_of_light
    current_gauss = gauss_fit(s, crisp_recon_current, A=np.max(crisp_recon_current))

    profile = np.hstack((s.reshape(-1, 1), crisp_recon_current.reshape(-1, 1)))
    #bshape = BeamShape()
    #bshape.Imax = np.max(crisp_recon_current)
    #profile = bshape.get(250)

    wl = WakeLoss(wakefile="wake_undulator_OCELOT.txt")
    loss_factor = wl.get_loss_factor(profile)

    print(loss_factor, chirp)

    ax1 = plt.subplot(211)
    plt.plot(s * 1e6, profile[:, 1] * 1e-3)
    plt.plot(s* 1e6, current_gauss * 1e-3)
    plt.xlabel(r"s [$\mu$m]")
    plt.ylabel("I [kA]")
    plt.setp(ax1.get_xticklabels(), visible=False)
    wake = wl.get_wake(profile)
    wake_energy_loss = wl.get_energy_loss(profile)
    indx = wl.get_cm(profile)
    wake_energy_loss = np.round(wake_energy_loss * 1e-3, 1)
    print("energy loss: ", wake_energy_loss * 6.1 * 21 * 1e-3, " MeV")
    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(wake[:, 0] * 1e6, wake[:, 1] * 1e-3, label="Wake")
    # plt.plot(wake[indx, 0]*1e6, wake[indx, 1] * 1e-3, "ro", lw=6, label="Loss: " + str(wake_energy_loss) + " keV")
    plt.ylabel("Wake [kV]")
    plt.xlabel(r"s [$\mu$m]")
    # plt.legend()
    plt.show()

    LOSS.append(loss_factor)
    CHIRP.append(chirp)
    #ax.plot(freq*1e-12, ff, label="L2 chirp = " + str(chirp))
    #ax.plot(freq * 1e-12, ff_noise, label="L2 chirp = " + str(chirp))
    #ax.plot(freq * 1e-12, np.sqrt(np.std(FF_all, axis=0)), label="L2 chirp = " + str(chirp))
    #ax1.plot(freq * 1e-12, det_lim, label="L2 chirp = " + str(chirp))
    #ax.plot(newfreqs * 1e-12, newff, label="L2 chirp = " + str(chirp))
    #ax.fill_between(freq*1e-12, np.zeros(np.size(det_lim)), y2=det_lim, label='Noise Floor', alpha=0.3,color='gray')



ax2.loglog()
ax2.legend()
ax1.legend()
#ax1.loglog()
#ax2.set_xlabel("f [THz]")
#ax2.set_ylabel(r"$|FF|^2$")
#plt.ylim(0.001, 10)
#plt.loglog()

x = np.array([data["chirp"] for data in DATA2])

plt.show()

plt.figure(3)
bcm1_raw = np.hstack((np.array([data[BCM_B2_ch] for data in DATA1]), np.array([data[BCM_B2_ch] for data in DATA2])))
bcm2_raw = np.hstack((np.array([data[BCM_B2_2_ch] for data in DATA1]), np.array([data[BCM_B2_2_ch] for data in DATA2])))
bcm_bc1_raw = np.hstack((np.array([data[BCM1_B1] for data in DATA1]), np.array([data[BCM1_B1] for data in DATA2])))

data["bcm1"] = bcm1_raw
data["bcm2"] = bcm2_raw
data["bcm_bc1"] = bcm_bc1_raw

plt.errorbar(x, np.mean(bcm1_raw, axis=1), yerr=np.std(bcm1_raw, axis=1), label="BCM1")
plt.errorbar(x, np.mean(bcm2_raw, axis=1), yerr=np.std(bcm1_raw, axis=1), label="BCM2")
#A = np.vstack([x, np.ones(len(x))]).T
#m, c = np.linalg.lstsq(A, losses, rcond=None)[0]
#print("A, B  = ", m, c)

#plt.plot(x, m*x + c, ".-", label="lin reg: " + str(np.round(m, 2)) + " MeV / step")
#plt.ylabel("BCM [a.u.]")
#plt.yticks(np.linspace(losses.min(), losses.max(), num=5))

#plt.xlabel(r"L1 Chirp [a.u.]")
plt.legend()




energy_T4_raw = np.hstack((np.array([data[T4_energy_ch] for data in DATA1]), np.array([data[T4_energy_ch] for data in DATA2])))
energy_T4D_raw = np.hstack((np.array([data[T4D_energy_ch] for data in DATA1]), np.array([data[T4D_energy_ch] for data in DATA2])))
data["t4"] = energy_T4_raw
data["t4d"] = energy_T4D_raw

energy_T4_mean = np.mean(energy_T4_raw, axis=1)
energy_T4_std = np.std(energy_T4_raw, axis=1)
energy_T4D_mean = np.mean(energy_T4D_raw, axis=1)
energy_T4D_std = np.std(energy_T4D_raw, axis=1)

CALIB_COEF =1.99/1.27


plt.figure(4)
#plt.title("Energy meas. in T4 and T4D stations. Run #1")
#plt.errorbar(x, energy_T4, yerr=t4_std, label="T4 energy meas.")
#plt.errorbar(x, -(energy_T4_mean - energy_T4_mean[0]) , yerr=energy_T4_std, label="T4 energy meas.")
plt.errorbar(x, -(energy_T4D_mean - energy_T4D_mean[0]) / CALIB_COEF/6.1/21*1000, yerr=energy_T4D_std/CALIB_COEF/6.1/21*1000)
plt.plot(CHIRP, -np.array(LOSS)*1e-3)

#A = np.vstack([x, np.ones(len(x))]).T
#m, c = np.linalg.lstsq(A, losses, rcond=None)[0]
#print("A, B  = ", m, c)

#plt.plot(x, m*x + c, ".-", label="lin reg: " + str(np.round(m, 2)) + " MeV / step")
#plt.ylabel("Energy Loss [MeV]")
#plt.yticks(np.linspace(losses.min(), losses.max(), num=5))

#plt.xlabel(r"L1 Chirp")
#plt.legend()

#tld = np.array([data[TLD_energy_ch + ".MEAN"] for data in DATA1])
cl = np.array([data[CL_energy_ch + ".MEAN"] for data in DATA1])
t4 = np.array([data[T4_energy_ch + ".MEAN"] for data in DATA1])
plt.figure(50)
plt.title("Energy meas. in T4/TLD/CL stations. Run #1")
#plt.errorbar(x, energy_T4, yerr=t4_std, label="T4 energy meas.")
#plt.errorbar(x, -(energy_T4_mean - energy_T4_mean[0]) , yerr=energy_T4_std, label="T4 energy meas.")
#plt.plot(x, tld, label="TLD")
plt.plot(x, t4, label="T4")
plt.plot(x, cl, label="CL")
plt.legend()
plt.show()



#pickle.dump( data, open( "LH_4000.p", "wb" ) )