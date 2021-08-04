import numpy as np
import pandas as pd
import pickle
from scipy.interpolate import interp1d
import simulate_spectrometer_signal as crisp
import reconstruction_module as crisp_recon
import matplotlib.pyplot as plt

def interpolate_data(slist, profiles, p_moments):
    """

    """
    s_max = np.max([np.max(s) for s in slist])
    s_min = np.min([np.min(s) for s in slist])
    # s_max = 110e-6
    # s_min = -110e-6
    sq = np.linspace(s_min, s_max, num=1024)
    print(f"s_min = {s_min}, s_max = {s_max}")
    prof_interp = []
    mp_interp = []
    for s, prof, mp in zip(slist, profiles, p_moments):
        f = interp1d(s, prof, bounds_error=False, fill_value=0.)
        profile = f(sq)
        prof_interp.append(profile)

        fmp = interp1d(s, mp, bounds_error=False, fill_value=0.)
        mp_sq = fmp(sq)
        mp_interp.append(mp_sq)
    return sq, prof_interp, mp_interp


def filter_and_recollect(db):
    rf = []
    cavity = []
    slist = []
    profiles = []
    ffactors = []
    freqs = []
    p_moments = []
    s1 = []
    s2 = []
    # read all databases
    for d in db:
        I = d['I']
        if np.max(I) < 2000:
            continue
        s = d["s"]

        # centered the current profile
        cm = np.trapz(I * s, s) / np.trapz(I, s)
        s_shift = s - cm
        s1.append(np.min(s_shift))
        s2.append(np.max(s_shift))
        slist.append(s_shift)

        profiles.append(I)

        # generate new ffactor
        freq, ff, ff_noise, det_lim = crisp.get_crisp_signal(s_shift, I, n_shots=1000, which_set='both')
        # *****************************************************************
        # for real CRISP signal you need to -     ff == np.sqrt(FF_crisp_real)
        # new_freqs, final_ff, final_ff_noise = crisp_recon.cleanup_formfactor(freq, ff)
        # new_freqs_CRISP, final_ff_CRISP, final_ff_noise_CRISP = crisp_recon.cleanup_formfactor(freq_CRISP, np.sqrt(FF_crisp_real))
        # *****************************************************************


        freqs.append(freq)
        ffactors.append(ff)

        rf.append(np.array([d['chirp'], d['curv'], d['skew'] * 1e3, d['l1'], d['l2']]))
        cavity.append(np.array([d["A1.v"], d["A1.phi"], d["AH1.v"], d["AH1.phi"],
                                d["L1.v"], d["L1.phi"], d["L2.v"], d["L2.phi"]]))

        p_moments.append(d["mp"])

    sq, prof_interp, mp_interp = interpolate_data(slist, profiles, p_moments)

    # new_db = {"s": sq, "profiles": prof_interp, "mp":mp_interp, "freq":freqs[0],
    #          "ff": ffactors, "rf": rf, "cavity":cavity}
    new_db = {"s": sq, "s1": s1, "s2": s2, "profiles": profiles, "mp": mp_interp, "rf": rf,
              "cavity": cavity, "ff": ffactors, "freq": freqs[0]}
    #with open('sim_over_.p', 'wb') as handle:
    #    pickle.dump(new_db, handle)
    return new_db


db = pickle.load(open('gather_80k.p', "rb"))
print("Number of simulations: ", len(db))

# take one simulation
one_sim = db[2681]

print("Saved parameters: ", list(one_sim.keys()))
print(f"Injector RF chirp = {one_sim['chirp']}")
print(f"Injector RF curv = {one_sim['curv']}")
print(f"Injector RF skew = {one_sim['skew']*1e3}")
print(f"L1 RF chirp = {one_sim['l1']}")
print(f"L2 RF chirp = {one_sim['l2']}")
print()
print("RF params in terms of phases and amplitudes:")
print(f"A1 V = {one_sim['A1.v']} MV")
print(f"A1 phi = {one_sim['A1.phi']} deg")
print(f"AH1 V = {one_sim['AH1.v']} MV")
print(f"AH1 phi = {one_sim['AH1.phi']} deg")
print(f"L1 V = {one_sim['L1.v']} MV")
print(f"L1 phi = {one_sim['L1.phi']} deg")
print(f"L2 V = {one_sim['L2.v']} MV")
print(f"L2 phi = {one_sim['L2.phi']} deg")

plt.figure(1)
plt.title("Current profile")
plt.plot(one_sim["s"] * 1e6, one_sim["I"])
plt.xlabel("s [um]")
plt.ylabel("I [A]")

plt.figure(2)
plt.title("Slice mean energy")
plt.plot(one_sim["s"] * 1e6, one_sim["mp"])
plt.xlabel("s [um]")
plt.ylabel("mean energy")

plt.figure(3)
plt.title("CRISP signal")
plt.plot(one_sim["freq"]*1e-12, one_sim["formfactor"])
plt.loglog()
plt.xlabel("freq [THz]")
plt.ylabel("|FF|")

plt.show()

# Lets make DataFrame from it for current profiles more than 2000 A

df = pd.DataFrame(db)
df["Imax"] = df['I'].apply(np.max)

df_high = df[df["Imax"]>2000]
print("number of sim with curent > 2000:", len(df_high))


# interpolate current profiles
s_interp, I_interp, mp_interp = interpolate_data(df_high["s"], df_high["I"], df_high["mp"])
print(s_interp.shape)
print(len(I_interp))

