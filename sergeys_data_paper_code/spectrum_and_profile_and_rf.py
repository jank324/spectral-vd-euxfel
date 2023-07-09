import pickle

import matplotlib.pyplot as plt
import numpy as np
from reconstruction_module import *

speed_of_light = 299792458.0  # m/s


def reconstruction(freq, FF_list_crisp):
    ff_mean = np.mean(FF_list_crisp, axis=0)
    ff_mean[ff_mean < 0] = 0
    ff = np.sqrt(ff_mean)

    n_shots = FF_list_crisp.shape[0]
    print(n_shots)

    ff_noise, det_lim = get_noise_detlim(final_ff=ff, charge=250e-12, n_shots=n_shots)
    noise = np.sqrt(np.std(FF_list_crisp, axis=0))
    crisp_recon_time, crisp_recon_current, crisp_t_rms = master_recon(
        freq,
        ff,
        noise,
        det_lim,
        charge=0.25e-9,
        channels_to_remove=[],
        show_plots=False,
    )
    return crisp_recon_time, crisp_recon_current


data_lh_off = pickle.load(open("LH_OFF.p", "rb"))
data_lh_on = pickle.load(open("LH_4000.p", "rb"))


fig, (ax1, ax2) = plt.subplots(2, 1)
for i, chirp in enumerate(data_lh_off["chirp"]):
    freq_on = data_lh_on["freq"][i]
    FF_on = data_lh_on["FF"]
    recon_time_on, recon_current_on = reconstruction(freq_on, FF_list_crisp=FF_on[i])
    s = recon_time_on * speed_of_light
    ax1.plot(s, recon_current_on)

    freq_off = data_lh_off["freq"][i]
    FF_off = data_lh_off["FF"]

    recon_time_off, recon_current_off = reconstruction(
        freq_off, FF_list_crisp=FF_off[i]
    )

    s = recon_time_off * speed_of_light
    ax2.plot(s, recon_current_off)

plt.xlabel("t [fs]")
ax1.set_ylabel("LH ON: I [A]")
ax2.set_ylabel("LH OFF: I [A]")


plt.show()

rf_params = pickle.load(open("rf_params.p", "rb"))
key_data = [  # Injector - stays constant during scan
    "XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.PHASE",  # A1 phase
    "XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.AMPL",  # A1 ampl
    "XFEL.RF/LLRF.CONTROLLER/CTRL.AH1.I1/SP.PHASE",  # AH1 phase
    "XFEL.RF/LLRF.CONTROLLER/CTRL.AH1.I1/SP.AMPL",  # AH1 ampl
    # L1 linac - this is what we scanned
    "XFEL.RF/LLRF.CONTROLLER/CTRL.A2.L1/SP.PHASE",  # A2 phase
    "XFEL.RF/LLRF.CONTROLLER/CTRL.A2.L1/SP.AMPL",  # A2 ampl
    # L2 linac - stays constant during scan
    "XFEL.RF/LLRF.CONTROLLER/CTRL.A3.L2/SP.PHASE",  # A3 phase
    "XFEL.RF/LLRF.CONTROLLER/CTRL.A3.L2/SP.AMPL",  # A3 ampl
    "XFEL.RF/LLRF.CONTROLLER/CTRL.A4.L2/SP.PHASE",  # A4 phase
    "XFEL.RF/LLRF.CONTROLLER/CTRL.A4.L2/SP.AMPL",  # A4 ampl
    "XFEL.RF/LLRF.CONTROLLER/CTRL.A5.L2/SP.PHASE",  # A5 phase
    "XFEL.RF/LLRF.CONTROLLER/CTRL.A5.L2/SP.AMPL",  # A5 ampl
    # This is what you probably used in NN chirps and so on
    # injector we did not touch
    "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/I1/SUMVOLTAGE.AMPLITUDE.SP.1",  # injector amplitude
    "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/I1/SUMVOLTAGE.CHIRP.SP.1",  # injector chirp
    "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/I1/SUMVOLTAGE.CURVATURE.SP.1",  # injector second derivative (curvature)
    "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/I1/SUMVOLTAGE.THIRDDERIVATIVE.SP.1",  # injector third derivative (skew parameter)
    # L1 this is what we scanned
    "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L1/SUMVOLTAGE.AMPLITUDE.SP.1",  # L1 amplitude
    "chirp",  # L1 chirp
    # L2 did not touch
    "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L2/SUMVOLTAGE.AMPLITUDE.SP.1",  # L2 amplitude
    "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L2/SUMVOLTAGE.CHIRP.SP.1",  # L2 amplitude
]

for i, chirp in enumerate(rf_params["chirp"]):
    print(
        "L1: V ="
        f" {rf_params['XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L1/SUMVOLTAGE.AMPLITUDE.SP.1'][i]},"
        f" chirp = {rf_params['chirp'][i]}"
    )

# show only one others are the same
for i, chirp in enumerate(rf_params["chirp"]):
    print(
        "Injector: V ="
        f" {rf_params['XFEL.RF/LLRF.SUMVOLTAGE_CTRL/I1/SUMVOLTAGE.AMPLITUDE.SP.1'][i]}, "
        "chirp ="
        f" {rf_params['XFEL.RF/LLRF.SUMVOLTAGE_CTRL/I1/SUMVOLTAGE.CHIRP.SP.1'][i]}, "
        "curv ="
        f" {rf_params['XFEL.RF/LLRF.SUMVOLTAGE_CTRL/I1/SUMVOLTAGE.CURVATURE.SP.1'][i]}, "
        "skew ="
        f" {rf_params['XFEL.RF/LLRF.SUMVOLTAGE_CTRL/I1/SUMVOLTAGE.THIRDDERIVATIVE.SP.1'][i]}, "
    )
    print("Injector: Others are the same. break ... \n")
    break

for i, chirp in enumerate(rf_params["chirp"]):
    print(
        "L1: V ="
        f" {rf_params['XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L2/SUMVOLTAGE.AMPLITUDE.SP.1'][i]},"
        " chirp ="
        f" {rf_params['XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L2/SUMVOLTAGE.CHIRP.SP.1'][i]}"
    )
    print("Injector: Others are the same. break ... \n")
    break
