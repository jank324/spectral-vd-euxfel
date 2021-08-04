#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 07:24:42 2019

@author: xfeloper
"""
import sys
sys.path.append("/home/xfeloper/user/tomins/ocelot_test/pyBigBro")
import pydoocs
import numpy as np
import matplotlib.pyplot as plt
import time
from mint.machine import Machine, MPS
from mint.snapshot import Snapshot, SnapshotDB
import md_config_2020_26_bcm as conf
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

l1_chirp_ch = "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L1/SUMVOLTAGE.CHIRP.SP.1"
l1_chirp_0 = pydoocs.read(l1_chirp_ch)["data"]

#l2_chirp_ch = "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/I1/SUMVOLTAGE.CHIRP.SP.1"
#l2_chirp_0 = pydoocs.read(l2_chirp_ch)["data"]



print(l1_chirp_ch, " <-- ", l1_chirp_0)

#%%
# l2_chirp_phase_range = np.linspace(-8., 14,  0.5)
l1_chirp_phase_range = [ -11.5, -11, -10.5,-10, -9.5, -9, -8.5, -8, -7.5, -7, -6, -5, -3, -1, +2, +6 ]

snapshot = conf.snapshot

machine = Machine(snapshot)
db = SnapshotDB(filename = time.strftime("%Y%m%d-%H_%M_%S")+ "_scan_TDS_ph_minus_3" + ".pcl")
df_ref = machine.get_machine_snapshot()
db.add(df_ref)

time.sleep(1)

take_background(db, machine)
time.sleep(1)


for x in [0]:
    print("{} <-- {}".format("something", x))
    # pydoocs.write(a1_ph_ch, a1_ph)
    #
    time.sleep(0.5)
    for l1_chirp in l1_chirp_phase_range:
        print("{} <-- {}".format(l1_chirp_ch, l1_chirp))
        pydoocs.write(l1_chirp_ch, l1_chirp)
        print("sleep 5 sec" )
        time.sleep(1)
        
        while True:
            if machine.is_machine_online():
                break
            else:
                time.sleep(3)
                print("sleep 3 sec ..")

        for i in range(10):
            print(f"taking image {i}")
            df = machine.get_machine_snapshot()
            db.add(df)
            time.sleep(1)
db.save()


#pydoocs.write(l1_chirp_ch, l1_chirp_0)


