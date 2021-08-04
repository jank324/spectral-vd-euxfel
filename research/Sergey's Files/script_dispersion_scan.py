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


# High dispersion optics
quads = ["XFEL.MAGNETS/MAGNET.ML/QI.52.I1/KICK_MRAD.SP",
"XFEL.MAGNETS/MAGNET.ML/QI.53.I1/KICK_MRAD.SP",
"XFEL.MAGNETS/MAGNET.ML/QI.54.I1/KICK_MRAD.SP",
"XFEL.MAGNETS/MAGNET.ML/QI.55.I1/KICK_MRAD.SP",
"XFEL.MAGNETS/MAGNET.ML/QI.57.I1/KICK_MRAD.SP",
"XFEL.MAGNETS/MAGNET.ML/QI.59.I1/KICK_MRAD.SP",
"XFEL.MAGNETS/MAGNET.ML/QI.60.I1/KICK_MRAD.SP",
"XFEL.MAGNETS/MAGNET.ML/QI.61.I1/KICK_MRAD.SP",
"XFEL.MAGNETS/MAGNET.ML/QI.63.I1D/KICK_MRAD.SP",
"XFEL.MAGNETS/MAGNET.ML/QI.64.I1D/KICK_MRAD.SP"]


kk12 = [-83.7203, 500.308, 188.82, -712.48, 712.48,-712.48, -509.559, 832.63, -249.585, 831.95]

# set 
for i, q in enumerate(quads):
    print("set basic optics = ",q, " <--", kk12[i])
    pydoocs.write(q, kk12[i])




quads = ["XFEL.MAGNETS/MAGNET.ML/QI.60.I1/KICK_MRAD.SP",
		"XFEL.MAGNETS/MAGNET.ML/QI.61.I1/KICK_MRAD.SP",
		"XFEL.MAGNETS/MAGNET.ML/QI.63.I1D/KICK_MRAD.SP",
		"XFEL.MAGNETS/MAGNET.ML/QI.64.I1D/KICK_MRAD.SP"]
		
kk12 = [-509.559, 832.63, -249.585, 831.95]
kk10 = [-509.0345, 832.175, 106.965, 475.4]
kk08 = [-508.965, 820.2076, 582.365, 0]
kk06 = [-508.749, 789.625, 1046.306, -475.4]


for i, q in enumerate(quads):
    print("set = ", kk12[i])
    #pydoocs.write(q, kk12[i])
    
#exit()

snapshot = conf.snapshot

machine = Machine(snapshot)
db = SnapshotDB(filename = time.strftime("%Y%m%d-%H_%M_%S")+ "_Dx_1170_tds_18" + ".pcl")
df_ref = machine.get_machine_snapshot()
db.add(df_ref)

time.sleep(1)

take_background(db, machine)
time.sleep(1)



for i in range(15):

    print(f"taking image {i}")
    df = machine.get_machine_snapshot(check_if_online=True)
    db.add(df)
    time.sleep(1)

db.save()




