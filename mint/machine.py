# -*- coding: utf-8 -*-
"""
S.Tomin

Machine class to get machine snapshot
"""

from mint.xfel_interface import XFELMachineInterface
from mint.devices import Device
import pandas as pd 
import numpy as np
import time
from datetime import datetime
import imageio
import matplotlib

import os
import pickle


class MPS(Device):
    def __init__(self, eid=None, server="XFEL", subtrain="SA1"):
        super(MPS, self).__init__(eid=eid)
        self.mi = XFELMachineInterface()
        self.subtrain = subtrain
        self.server = server

    def beam_off(self):
        self.mi.set_value(self.server + ".UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED", 0)

    def beam_on(self):
        self.mi.set_value(self.server + ".UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED", 1)

    def num_bunches_requested(self, num_bunches=1):
        self.mi.set_value(self.server + ".UTIL/BUNCH_PATTERN/CONTROL/NUM_BUNCHES_REQUESTED_1", num_bunches)
    
    def is_beam_on(self):
        val = self.mi.get_value(self.server + ".UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED")
        return val


class Machine:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.mi = XFELMachineInterface()
        self.bpm_server = "ORBIT"     # or "BPM"
        self.server = "XFEL"
        self.subtrain = "ALL"
        self.suffix = ""


    def is_machine_online(self):
        """
        method to check if machine is online

        :return: True if online
        """

        alarm_status = []
        for i, alarm in enumerate(self.snapshot.alarm_channels):
            try:
                val = self.mi.get_value(alarm)
            except Exception as e:
                print("id: " + alarm+ " ERROR: " + str(e))
                val = 0
            min_val, max_val = self.snapshot.alarm_bounds[i]
            if min_val < val < max_val:
                alarm_status.append(True)
            else:
                alarm_status.append(False)
        if not np.array(alarm_status).all():
            return False
        return True

    def get_orbit(self, data, all_names):
        for sec_id in self.snapshot.orbit_sections:
            try:
                orbit_x = np.array(self.mi.get_value(self.server + ".DIAG/" + self.bpm_server + "/*." + sec_id + "/X." + self.subtrain + self.suffix))

                orbit_y = np.array(self.mi.get_value(self.server + ".DIAG/" + self.bpm_server + "/*." + sec_id + "/Y."  + self.subtrain + self.suffix))
            except Exception as e:
                print("orbit id: " + sec_id+ " ERROR: " + str(e))
                return [], []
            x = orbit_x[:, 1].astype(np.float)
            y = orbit_y[:, 1].astype(np.float)
            xy = np.append(x, y)

            names_x =  [name + ".X" for name in orbit_x[:, 4]]
            names_y =  [name + ".Y" for name in orbit_y[:, 4]]
            names = np.append(names_x, names_y)
            data = np.append(data, xy)
            all_names = np.append(all_names, names)
        return data, all_names

    def get_magnets(self, data, all_names):
        for sec_id in self.snapshot.magnet_sections:
            try:
                magnets = np.array(self.mi.get_value("XFEL.MAGNETS/MAGNET.ML/*." + sec_id + "/KICK_MRAD.SP"))
            except Exception as e:
                print("magnets id: " + sec_id+ " ERROR: " + str(e))
                return [], []
            vals = magnets[:, 1].astype(np.float)

            names = [name for name in magnets[:, 4]]
            data = np.append(data, vals)
            all_names = np.append(all_names, names)
        return data, all_names

    def get_phase_shifters(self, data, all_names):
        for sec_id in self.snapshot.phase_shifter_sections:
            try:
                phase_shifters = np.array(self.mi.get_value("XFEL.FEL/WAVELENGTHCONTROL." + sec_id + "/BPS.*." + sec_id + "/GAP.OFFSET"))
            except Exception as e:
                print("ERROR: magnets: ", sec_id, e)
                return [], []
            vals = phase_shifters[:, 1].astype(np.float)

            names = [name for name in phase_shifters[:, 4]]
            data = np.append(data, vals)
            all_names = np.append(all_names, names)
        return data, all_names

    def get_undulators(self, data, all_names):
        for sec_id in self.snapshot.undulators:
            try:
                if sec_id in ["SA1", "SA2"]:
                    undulators = np.array(self.mi.get_value("XFEL.FEL/WAVELENGTHCONTROL." + sec_id + "/U40.*." + sec_id + "/GAP"))
                else:
                    undulators = np.array(
                        self.mi.get_value("XFEL.FEL/WAVELENGTHCONTROL." + sec_id + "/U68.*." + sec_id + "/GAP"))
            except Exception as e:
                print("ERROR: magnets: ", sec_id, e)
                return [], []
            vals = undulators[:, 1].astype(np.float)

            names = [name for name in undulators[:, 4]]
            data = np.append(data, vals)
            all_names = np.append(all_names, names)
        return data, all_names


    def get_channels(self, data, all_names):
        data = list(data)
        for ch in self.snapshot.channels:
            try:
                val = self.mi.get_value(ch)
            except Exception as e:
                print("id: " + ch + " ERROR: " + str(e))
                val = np.nan
            data.append(val)
            all_names = np.append(all_names, ch)
        return data, all_names

    def get_images(self, data, all_names):
        for i, ch in enumerate(self.snapshot.images):
            folder = self.snapshot.image_folders[i]
            try:
                img = self.mi.get_value(ch)
            except Exception as e:
                print("id: " + ch + " ERROR: " + str(e))
                img = None

            cam_name = ch.split("/")[-2]
            #datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            #name = cam_name + "-" + time.strftime("%Y%m%d-%H%M%S")
            name = cam_name + "-" + datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')[:-3]

            filename = name + ".png"
            path = folder + os.sep + filename
            path_pcl = folder + os.sep + name + ".pcl"
            if img is not None:
                # scipy.misc.imsave(path, img)
                #imageio.imwrite(path, img)
                matplotlib.image.imsave(path, img)
                with open(path_pcl, 'wb') as f:
                    # Pickle the 'data' dictionary using the highest protocol available.
                    pickle.dump(img, f)

            else:
                path = None
            #print(data)
            #data = list(data)
            #data = np.append(data, np.array([path], dtype=object))
            data.append(path)
            all_names = np.append(all_names, ch)
        return data, all_names

    def wait_machine_online(self):
    
        if self.is_machine_online():
            return 
            
        while True:
            if self.is_machine_online():
                print("machine is BACK. Wait 5 sec for recovering and continue")
                time.sleep(5)
                break
            else:
                time.sleep(3)
                print("machine is OFFLINE. Sleep 3 sec ..")

    def get_machine_snapshot(self, check_if_online=False):

        #if not self.is_machine_online():
        #    print("machine is not online. wait 3 sec ...")
        #    time.sleep(2)
        #    return None
        if check_if_online:
        
            self.wait_machine_online()

        data = np.array([time.time()], dtype=object)
        all_names = np.array(["timestamp"])
        data, all_names = self.get_orbit(data, all_names)
        if len(data) == 0:
            return None
        data, all_names = self.get_magnets(data, all_names)
        if len(data) == 0:
            return None
        data, all_names = self.get_phase_shifters(data, all_names)
        if len(data) == 0:
            return None
        data, all_names = self.get_undulators(data, all_names)
        if len(data) == 0:
            return None
        data, all_names = self.get_channels(data, all_names)
        if len(data) == 0:
            return None
        data, all_names = self.get_images(data, all_names)
        if len(data) == 0:
            print("get_images bad")
            return None
        #print(len(data), len(all_names))
        data_dict = {}
        for name, d in zip(all_names, data):
            data_dict[name] = [d]

        df = pd.DataFrame(data_dict, columns=data_dict.keys())
        return df
            
    

if __name__ is "__main__":
    df = pd.DataFrame()
    df2 = pd.DataFrame(data = {"a": 2, "b":4}, index=[0])
    df3 = pd.DataFrame(data = {"b": [5], "a":[6], "c": [3]})
    df = df.append(df2)
    print("1", df)
    df = df.append(df3)
    print("2", df)