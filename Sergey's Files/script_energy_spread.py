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
        


snapshot = conf.snapshot

machine = Machine(snapshot)
db = SnapshotDB(filename = time.strftime("%Y%m%d-%H_%M_%S")+ "_80_MeV_100pC" + ".pcl")
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




