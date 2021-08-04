import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


font = {#'family' : 'normal',
    #'weight' : 'bold',
    'size'   : 16}

matplotlib.rc('font', **font)

filename = "20210330-16_53_40660eV_control_calib.p"
#filename = "20210330-19_20_18660eV_control_calib.p"
T4D_energy_ch = "XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/T4D/ENERGY.ALL"
DATA2 = pickle.load(open(filename, "rb"))

# CHanneld to PLOT
energy_ch_plot = T4D_energy_ch

energy = np.array([data[energy_ch_plot + ".MEAN"] for data in DATA2])
print(energy_ch_plot, energy)
energy_std = np.array([np.std(data[energy_ch_plot]) for data in DATA2])

losses = -(energy - energy[0])

x = np.arange(len(energy))
# plt.errorbar(x, losses, yerr=energy_std)
# plt.xlabel("N undulator")
# plt.ylabel("Energy loss [MeV]")
# plt.show()

SASE3_Eph = 660

# ax = plt.subplot(211)

#plt.title("SASE3 SR losses against closed undulator cells. Eph = " + str(np.round(SASE3_Eph, 1)))
plt.errorbar(x*3, losses, yerr=energy_std, label="measurement")
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, losses, rcond=None)[0]
print("A, B  = ", m, c)

plt.plot(x*3, m * x + c, ".-", label="lin. reg: " + str(np.round(m/3, 2)) + " MeV/cell")
#plt.ylabel("Energy loss [MeV]")
# plt.yticks(np.linspace(losses.min(), losses.max(), num=5))

#plt.xlabel(r"Step number")
plt.legend()
plt.show()
