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
import config_L1_coupler_kick as conf
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
    time.sleep(1)
        

#l2_chirp_ch = "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L2/SUMVOLTAGE.CHIRP.SP.1"
#l2_chirp_0 = pydoocs.read(l2_chirp_ch)["data"]

#l1_chirp_ch = "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L1/SUMVOLTAGE.CHIRP.SP.1"
#l1_chirp_0 = pydoocs.read(l1_chirp_ch)["data"]

#i1_chirp_ch = "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/I1/SUMVOLTAGE.CHIRP.SP.1"
#i1_chirp_0 = pydoocs.read(i1_chirp_ch)["data"]

#i1_phase_ch = "XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.PHASE"
#i1_phase_0 = pydoocs.read(l1_phase_ch)["data"]

#i1_amp_ch = "XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.AMPL"
#i1_amp_0 = pydoocs.read(l1_amp_ch)["data"]


l1_phase_ch = "XFEL.RF/LLRF.CONTROLLER/CTRL.A2.L1/SP.PHASE"
l1_phase_0 = pydoocs.read(l1_phase_ch)["data"]

l1_amp_ch = "XFEL.RF/LLRF.CONTROLLER/CTRL.A2.L1/SP.AMPL"
l1_amp_0 = pydoocs.read(l1_amp_ch)["data"]



A = l1_amp_0/np.cos(l1_phase_0*np.pi/180)


#%%
# l2_chirp_phase_range = np.linspace(-8., 14,  0.5)
l1_phase_range = np.arange(-10., 11, 2.)

snapshot = conf.snapshot

machine = Machine(snapshot)
db = SnapshotDB(filename = time.strftime("%Y%m%d-%H_%M_%S")+ "_L1_phase_no_comp" + ".pcl")
df_ref = machine.get_machine_snapshot()
db.add(df_ref)

time.sleep(1)

#take_background(db, machine)


for x in [0]:
    print("{} <-- {}".format("something", x))
    # pydoocs.write(a1_ph_ch, a1_ph)
    #
    time.sleep(0.5)
    for l1_phase in l1_phase_range:
        print("{} <-- {}".format(l1_phase_ch, l1_phase))
        A_new = A/np.cos(l1_phase*np.pi/180)
        A_to_set = A*np.cos(l1_phase*np.pi/180)
        print("{} <-- {}".format(l1_amp_ch, A_to_set))
        #pydoocs.write(l1_phase_ch, l1_phase)
        pydoocs.write(l1_amp_ch, A_to_set)
        time.sleep(0.2)
        #pydoocs.write(l1_phase_ch, l1_phase)
        pydoocs.write(l1_amp_ch, A_to_set)
        print("sleep 2 sec" )
        time.sleep(1)
        
        while True:
            if machine.is_machine_online():
                break
            else:
                time.sleep(3)
                print("sleep 3 sec ..")

        for i in range(10):
            print(f"taking data {i}")
            df = machine.get_machine_snapshot()
            db.add(df)
            time.sleep(0.2)
db.save()


pydoocs.write(l1_phase_ch, l1_phase_0)
pydoocs.write(l1_amp_ch, l1_amp_0)


