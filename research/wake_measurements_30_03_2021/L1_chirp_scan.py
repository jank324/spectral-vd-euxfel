#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:25:39 2020

@author: xfeloper
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 07:04:33 2020

@author: xfeloper
"""

import numpy as np
import pydoocs
import matplotlib.pyplot as plt
import time
import pickle 
import matplotlib
font = {#'family' : 'normal',
    #'weight' : 'bold',
    'size'   : 16}

matplotlib.rc('font', **font)

#%%


def collect_stat_scalar(channels, timeout=0.2, nreads=10):
    data = {}
    for ch in channels:
        data[ch] = []
        
    
    for i in range(nreads):
        for ch in channels:
            v = pydoocs.read(ch)["data"]
            data[ch].append(v)
        
        time.sleep(timeout)
    data_mean = {}
    for ch in channels:
        data_mean[ch + ".MEAN"] = np.mean(data[ch])
    
    return data_mean, data




crisp_ch = 'XFEL.SDIAG/THZ_SPECTROMETER.FORMFACTOR/CRD.1934.TL/FORMFACTOR.XY'
crisp_int_ch = "XFEL.SDIAG/THZ_SPECTROMETER.FORMFACTOR/CRD.1934.TL/INTENSITY.TD"
crisp_status_ch = "XFEL.SDIAG/THZ_SPECTROMETER.GRATINGMOVER/CRD.1934.TL/STATUS.STR"
crisp_lim_ch = "XFEL.SDIAG/THZ_SPECTROMETER.FORMFACTOR/CRD.1934.TL/FORMFACTOR_DETECTLIMIT.XY"

BCM_B2_ch = "XFEL.FEEDBACK/FT1.LONGITUDINAL/MONITOR12/MEAN"
BCM_B2_2_ch = "XFEL.FEEDBACK/FT1.LONGITUDINAL/MONITOR13/MEAN"
BCM1 = "XFEL.SDIAG/BCM.ML/BCM.416.B2.1/BCM.TD"
BCM2 = "XFEL.SDIAG/BCM.ML/BCM.416.B2.2/BCM.TD"
BCM1_B1 = "XFEL.FEEDBACK/FT1.LONGITUDINAL/MONITOR8/MEAN"

CL_energy_ch = "XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/CL/ENERGY.ALL"
TLD_energy_ch = "XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/TLD/ENERGY.ALL"
T4_energy_ch = "XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/T4/ENERGY.ALL"
T4D_energy_ch = "XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/T4D/ENERGY.ALL"
#%%

L1_chirp_ch = "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L1/SUMVOLTAGE.CHIRP.SP.1"


energy_chls = [CL_energy_ch, T4_energy_ch, T4D_energy_ch, BCM_B2_ch, L1_chirp_ch, BCM_B2_2_ch, BCM1_B1]

DATA = []
data = {}

NREADS = 40

#L1_range = np.array([-11, -10.5, -10, -9.5, -9.25, -9,-8.75, -8.5, -8, -7.5, -7, -6.5, -6, -5, -4, -2, 0, 2, 5])

L1_range = np.array([-12, -11.5, -11, -10.5,-10, -9.5, -9, -8.5, -8, -7.5, -7, -6, -5, -3, -1, +2, +6 ])

for chirp in L1_range:
    data = {}

    pydoocs.write(L1_chirp_ch, chirp)
    print(f"SET chirp {chirp}, sleep 20 sec ... ")
    time.sleep(25)

    crisp_data = []
    crisp_status = []
    crisp_int = []
    crisp_lim = []
    for i in range(50):
        crisp_ff = pydoocs.read(crisp_ch)["data"]
        crisp_data.append(crisp_ff)
        crisp_int.append(pydoocs.read(crisp_int_ch)["data"])
        crisp_status.append(pydoocs.read(crisp_status_ch)["data"])
        crisp_lim.append(pydoocs.read(crisp_lim_ch)["data"])
        
        time.sleep(0.1)
        
    energy_mean, energy_data = collect_stat_scalar(energy_chls, timeout=0.2, nreads=NREADS)
    data.update(energy_mean)
    data.update(energy_data)
    data.update({"CRISP": crisp_data})
    data.update({"CRISP.INT": crisp_int})
    data.update({"CRISP.STS": crisp_status})
    data.update({"CRISP.LIM": crisp_lim})
    data.update({"chirp": chirp})
    DATA.append(data)


filename = time.strftime("%Y%m%d-%H_%M_%S")  + "_stage1_LH_4000.p"
pickle.dump(DATA, open(filename, "wb"))
#%%
DATA2 = pickle.load(open(filename, "rb"))



energy_T4 = np.array([data[T4_energy_ch + ".MEAN"] for data in DATA2])


energy_T4D = np.array([data[T4D_energy_ch + ".MEAN"] for data in DATA2])


losses = (energy_T4 - energy_T4D) - (energy_T4[0] - energy_T4D[0])
print(losses)
x = np.array([data["chirp"] for data in DATA2])


plt.figure(1)
plt.title("WAKE losses against L2 chirp (diff between T4 and T4D stations)")
plt.plot(x, losses, label="Loss")
#A = np.vstack([x, np.ones(len(x))]).T
#m, c = np.linalg.lstsq(A, losses, rcond=None)[0]
#print("A, B  = ", m, c)

#plt.plot(x, m*x + c, ".-", label="lin reg: " + str(np.round(m, 2)) + " MeV / step")
plt.ylabel("Energy loss [MeV]")
#plt.yticks(np.linspace(losses.min(), losses.max(), num=5))

plt.xlabel(r"L2 Chirp")
plt.legend()

    
    
# CHanneld to PLOT
t4_std = np.array([np.std(data[T4_energy_ch]) for data in DATA2])

energy_T4 = (energy_T4 - energy_T4[0])

t4d_std = np.array([np.std(data[T4D_energy_ch]) for data in DATA2])

energy_T4D = (energy_T4D - energy_T4D[0])


plt.figure(4)
plt.title("Energy meas. in T4 and T4D stations.")
plt.errorbar(x, energy_T4, yerr=t4_std, label="T4 energy meas.")
plt.errorbar(x, energy_T4D, yerr=t4d_std, label="T4D energy meas.")
#A = np.vstack([x, np.ones(len(x))]).T
#m, c = np.linalg.lstsq(A, losses, rcond=None)[0]
#print("A, B  = ", m, c)

#plt.plot(x, m*x + c, ".-", label="lin reg: " + str(np.round(m, 2)) + " MeV / step")
plt.ylabel("E - E0 [MeV]")
#plt.yticks(np.linspace(losses.min(), losses.max(), num=5))

plt.xlabel(r"L2 Chirp")
plt.legend()



# CHanneld to PLOT
bcm_std = np.array([np.std(data[BCM_B2_ch]) for data in DATA2])
bcm = np.array([data[BCM_B2_ch + ".MEAN"] for data in DATA2])


t4d_std = np.array([np.std(data[T4D_energy_ch]) for data in DATA2])

energy_T4D = (energy_T4D - energy_T4D[0])


plt.figure(6)
plt.title("Energy meas. in T4 and T4D stations.")
plt.errorbar(x, bcm, yerr=bcm_std, label="BCM")
#plt.plot(x, losses, label="Loss")
#A = np.vstack([x, np.ones(len(x))]).T
#m, c = np.linalg.lstsq(A, losses, rcond=None)[0]
#print("A, B  = ", m, c)

#plt.plot(x, m*x + c, ".-", label="lin reg: " + str(np.round(m, 2)) + " MeV / step")
plt.ylabel("BCM [a.u.]")
#plt.yticks(np.linspace(losses.min(), losses.max(), num=5))

plt.xlabel(r"L2 Chirp")
plt.legend()
plt.show()


