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

i1_chirp_ch = "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/I1/SUMVOLTAGE.CHIRP.SP.1"
i1_chirp_0 = pydoocs.read(i1_chirp_ch)["data"]

i1_curv_ch = "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/I1/SUMVOLTAGE.CURVATURE.SP.1"
i1_curv_0 = pydoocs.read(i1_curv_ch)["data"]

i1_skew_ch = "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/I1/SUMVOLTAGE.THIRDDERIVATIVE.SP.1"
i1_skew_0 = pydoocs.read(i1_skew_ch)["data"]

#i1_phase_ch = "XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.PHASE"
#i1_phase_0 = pydoocs.read(l1_phase_ch)["data"]

#i1_amp_ch = "XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.AMPL"
#i1_amp_0 = pydoocs.read(l1_amp_ch)["data"]


# A = l1_amp_0/np.cos(l1_phase_0*np.pi/180)


#%%
# l2_chirp_phase_range = np.linspace(-8., 14,  0.5)
chirp_range = np.arange(-5., 0,  1.)
curv_range = np.arange(0., 310.,  100.)
skew_range = np.arange(0., 40000.,  10000.)

print("chirp_range = ", chirp_range)
print("curv_range = ", curv_range)
print("skew_range = ", skew_range)

snapshot = conf.snapshot

machine = Machine(snapshot)
db = SnapshotDB(filename = time.strftime("%Y%m%d-%H_%M_%S")+ "_3Dscan_phase_2" + ".pcl")
df_ref = machine.get_machine_snapshot()
db.add(df_ref)

time.sleep(1)

take_background(db, machine)
time.sleep(1)


for chirp in chirp_range:
    print("{} <-- {}".format(i1_chirp_ch, chirp))
    pydoocs.write(i1_chirp_ch, chirp)
    time.sleep(0.1)
    pydoocs.write(i1_chirp_ch, chirp)
    time.sleep(1)
    
    for curv in curv_range:
        print("{} <-- {}".format(i1_curv_ch, curv))
        pydoocs.write(i1_curv_ch, curv)
        time.sleep(0.1)
        pydoocs.write(i1_curv_ch, curv)
        time.sleep(1)
        
        for skew in skew_range:
            print("{} <-- {}".format(i1_skew_ch, skew))
            pydoocs.write(i1_skew_ch, skew)
            time.sleep(0.1)
            pydoocs.write(i1_skew_ch, skew)
            print("sleep 2 sec" )
            time.sleep(1)
            
            while True:
                if machine.is_machine_online():
                    break
                else:
                    time.sleep(3)
                    print("sleep 3 sec ..")
    
            for i in range(3):
                print(f"taking image {i}")
                df = machine.get_machine_snapshot()
                db.add(df)
                time.sleep(1)
db.save()


pydoocs.write(i1_chirp_ch, i1_chirp_0)
pydoocs.write(i1_curv_ch, i1_curv_0)
pydoocs.write(i1_skew_ch, i1_skew_0)


