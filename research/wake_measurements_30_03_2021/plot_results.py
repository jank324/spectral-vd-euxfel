import numpy as np
import matplotlib.pyplot as plt
import pickle
from reconstruction_module import *
from ocelot.utils.wake_estimator import WakeLoss, BeamShape

speed_of_light = 299792458.0 # m/s


def reconstruction(freq, FF_list_crisp):
    ff_mean = np.mean(FF_list_crisp, axis=0)
    ff_mean[ff_mean < 0] = 0
    ff = np.sqrt(ff_mean)

    n_shots = FF_list_crisp.shape[0]

    ff_noise, det_lim = get_noise_detlim(final_ff=ff, charge=250e-12, n_shots=n_shots)
    noise = np.sqrt(np.std(FF_list_crisp, axis=0))
    crisp_recon_time, crisp_recon_current, crisp_t_rms = master_recon(freq, ff, noise,
                                                                      det_lim, charge=0.25e-9,
                                                                      channels_to_remove=[], show_plots=False)
    return crisp_recon_time, crisp_recon_current




CALIB_COEF =1.99/1.27

data_lh_off = pickle.load( open( "LH_OFF.p", "rb" ) )
data_lh_on = pickle.load( open( "LH_4000.p", "rb" ) )
print(data_lh_off["bcm1"].shape)



plt.figure(1)
plt.title("BCMs")
plt.errorbar(data_lh_off["chirp"], np.mean(data_lh_off["bcm1"], axis=1), yerr=np.std(data_lh_off["bcm1"], axis=1), label="BCM1: LH OFF")
plt.errorbar(data_lh_off["chirp"], np.mean(data_lh_off["bcm2"], axis=1), yerr=np.std(data_lh_off["bcm1"], axis=1), label="BCM2: LH OFF")
plt.errorbar(data_lh_on["chirp"], np.mean(data_lh_on["bcm1"], axis=1), yerr=np.std(data_lh_on["bcm1"], axis=1), label="BCM1: LH ON")
plt.errorbar(data_lh_on["chirp"], np.mean(data_lh_on["bcm2"], axis=1), yerr=np.std(data_lh_on["bcm1"], axis=1), label="BCM2: LH ON")
plt.xlabel("L1 chirp [a.u.]")
plt.ylabel("BCM [a.u.]")
plt.legend()

plt.figure(2)
plt.title("Energy Loss")
t4d_lh_off = np.mean(data_lh_off["t4d"], axis=1)/CALIB_COEF
plt.errorbar(data_lh_off["chirp"], -t4d_lh_off + t4d_lh_off[0], yerr=np.std(data_lh_off["t4d"], axis=1)/CALIB_COEF, label="T4D: LH OFF")
t4d_lh_on = np.mean(data_lh_on["t4d"], axis=1)/CALIB_COEF
plt.errorbar(data_lh_on["chirp"], -t4d_lh_on + t4d_lh_on[0], yerr=np.std(data_lh_on["t4d"], axis=1)/CALIB_COEF, label="T4D: LH ON")
plt.legend()
plt.xlabel("L1 chirp [a.u.]")
plt.ylabel("Energy Loss in SA3 undul. [MeV]")

plt.figure(3)
plt.title("Reconstructed Current")
LOSS_ON = []
CHIRP_ON = []
LOSS_OFF = []
CHIRP_OFF = []
for i, chirp in enumerate(data_lh_off["chirp"]):

    freq_on = data_lh_on["freq"][i]
    FF_on = data_lh_on["FF"]
    recon_time_on, recon_current_on = reconstruction(freq_on, FF_list_crisp=FF_on[i])
    s = recon_time_on*speed_of_light
    profile = np.hstack((s.reshape(-1, 1), recon_current_on.reshape(-1, 1)))
    wl = WakeLoss(wakefile="wake_undulator_OCELOT.txt")
    loss_factor = wl.get_loss_factor(profile)
    LOSS_ON.append(loss_factor)
    CHIRP_ON.append(data_lh_on['chirp'][i])
    print(i)
    freq_off = data_lh_off["freq"][i]
    FF_off = data_lh_off["FF"]

    recon_time_off, recon_current_off = reconstruction(freq_off, FF_list_crisp=FF_off[i])

    s = recon_time_off*speed_of_light
    profile = np.hstack((s.reshape(-1, 1), recon_current_off.reshape(-1, 1)))
    wl = WakeLoss(wakefile="wake_undulator_OCELOT.txt")
    loss_factor = wl.get_loss_factor(profile)
    LOSS_OFF.append(loss_factor)
    CHIRP_OFF.append(data_lh_off['chirp'][i])
    if -9. < chirp <-8.:
        plt.plot(recon_time_on * 1e15, recon_current_on, label=f"L1 chirp={data_lh_on['chirp'][i]}: LH ON")
        plt.plot(recon_time_off * 1e15, recon_current_off, label=f"L1 chirp={data_lh_off['chirp'][i]}: LH OFF")
plt.xlabel("t [fs]")
plt.ylabel("I [A]")
plt.legend()


plt.figure(20)
plt.title("Energy Loss")
t4d_lh_off = np.mean(data_lh_off["t4d"], axis=1)/CALIB_COEF
plt.errorbar(data_lh_off["chirp"], -t4d_lh_off + t4d_lh_off[0], yerr=np.std(data_lh_off["t4d"], axis=1)/CALIB_COEF, label="T4D: LH OFF")
plt.plot(CHIRP_OFF, -np.array(LOSS_OFF)*6.1*21*1e-6, label="LH OFF from recon")
t4d_lh_on = np.mean(data_lh_on["t4d"], axis=1)/CALIB_COEF
plt.errorbar(data_lh_on["chirp"], -t4d_lh_on + t4d_lh_on[0], yerr=np.std(data_lh_on["t4d"], axis=1)/CALIB_COEF, label="T4D: LH ON")
plt.plot(CHIRP_ON, -np.array(LOSS_ON)*6.1*21*1e-6, label="LH ON from recon")
plt.legend()
plt.xlabel("L1 chirp [a.u.]")
plt.ylabel("Energy Loss in SA3 undul. [MeV]")


plt.show()
