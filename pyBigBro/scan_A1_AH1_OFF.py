#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 07:24:42 2019

@author: xfeloper
"""
import sys
sys.path.append("/home/xfeloper/user/tomins/ocelot_new/pyBigBro")
import pydoocs
import numpy as np
import matplotlib.pyplot as plt
import time
from mint.machine import Machine, MPS
from mint.snapshot import Snapshot, SnapshotDB
import config_inj_study as conf
import time

def take_background(db, machine, nshots=5):
    print("background taking ... beam off")
    mps = MPS()
    mps.beam_off()
    time.sleep(1)
    for i in range(nshots):
        df = machine.get_machine_snapshot()
        db.add(df)
        time.sleep(1)
    print("background taking is over ... beam on")
    mps.beam_on()
        

#l2_chirp_ch = "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L2/SUMVOLTAGE.CHIRP.SP.1"
#l2_chirp_0 = pydoocs.read(l2_chirp_ch)["data"]

#l2_chirp_ch = "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L1/SUMVOLTAGE.CHIRP.SP.1"
#l2_chirp_0 = pydoocs.read(l2_chirp_ch)["data"]

#l2_chirp_ch = "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/I1/SUMVOLTAGE.CHIRP.SP.1"
#l2_chirp_0 = pydoocs.read(l2_chirp_ch)["data"]

i1_phase_ch = "XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.PHASE"
i1_phase_0 = pydoocs.read(i1_phase_ch)["data"]

i1_amp_ch = "XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.AMPL"
i1_amp_0 = pydoocs.read(i1_amp_ch)["data"]


A = i1_amp_0/np.cos(i1_phase_0*np.pi/180)


#%%
# l2_chirp_phase_range = np.linspace(-8., 14,  0.5)
i1_phase_range = np.arange(-6, 8,  2.)

snapshot = conf.snapshot

machine = Machine(snapshot)
db = SnapshotDB(filename = time.strftime("%Y%m%d-%H_%M_%S")+ "_q_500pC_run1_phase_2" + ".pcl")
df_ref = machine.get_machine_snapshot()
db.add(df_ref)

time.sleep(1)

take_background(db, machine)
time.sleep(1)

gun_phase_ch = "XFEL.RF/LLRF.CONTROLLER/CTRL.GUN.I1/SP.PHASE"
gun_phase_0 = pydoocs.read(gun_phase_ch)["data"]

for gun in [-49., -46., -43., -40., -37]:
    print("{} <-- {}".format("GUN phase", gun))
    pydoocs.write(gun_phase_ch, gun)
    time.sleep(0.2)
    pydoocs.write(gun_phase_ch, gun)
    time.sleep(10)
    for i1_phase in i1_phase_range:
        print("{} <-- {}".format(i1_phase_ch, i1_phase))
        A_new = A/np.cos(i1_phase*np.pi/180)
        print("{} <-- {}".format(i1_amp_ch, A_new))
        pydoocs.write(i1_phase_ch, i1_phase)
        pydoocs.write(i1_amp_ch, A_new)
        time.sleep(0.2)
        pydoocs.write(i1_phase_ch, i1_phase)
        pydoocs.write(i1_amp_ch, A_new)
        print("sleep 2 sec" )
        time.sleep(2)
        
        #while True:
        #    if machine.is_machine_online():
        #        break
        #    else:
        #        time.sleep(3)
        #        print("sleep 3 sec ..")

        for i in range(5):
            print(f"taking image {i}")
            df = machine.get_machine_snapshot(check_if_online=True)
            db.add(df)
            time.sleep(1)
db.save()


pydoocs.write(i1_phase_ch, i1_phase_0)
pydoocs.write(i1_amp_ch, i1_amp_0)
pydoocs.write(gun_phase_ch, gun_phase_0)

