# -*- coding: utf-8 -*-

"""
S.Tomin

 machine snapshot
"""
import pandas as pd 
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

class Snapshot():
    def __init__(self):
        self.sase_sections = ["SA1", "SA2", "SA3"]
        # or list of magnet prefix to check if they are vary
        self.magnet_prefix = None # ["QA.", "CBY.", "CAX.", "Q.", "CY.", "CX.", "CIX.", "CIY.", "QI.", "BL."]
        
        self.orbit_sections = {}
        self.magnet_sections = {}
        self.phase_shifter_sections = {}
        self.undulators = {}
        self.channels = []
        self.channels_tol = []

        # alarm channels
        self.alarm_channels = []
        self.alarm_bounds = []
        self.channels_track = []
        
        # multidim data
        self.images = []
        self.image_folders = []
    
    def add_alarm_channels(self, ch, min, max):
        """
        return scalar channels
        """
        self.alarm_channels.append(ch)
        self.alarm_bounds.append([min, max])
    
    def add_image(self, ch, folder):
        # check if folder exists and create if not
        if not os.path.exists(folder):
            os.makedirs(folder)
        if ch in self.images:
            print("WARNING: image channel is already added")
            return
        self.images.append(ch)
        self.image_folders.append(folder)
    
    def add_orbit_section(self, sec_id, tol=0.001, track=True):
        if sec_id in self.orbit_sections:
            print("WARNING: channel is already added")
            return
        self.orbit_sections[sec_id] = {"id": sec_id, "tol": tol, "track": track}

    def add_magnet_section(self, sec_id, tol=0.001, track=True):
        if sec_id in self.magnet_sections:
            print("WARNING: channel is already added")
            return
        self.magnet_sections[sec_id] = {"id": sec_id, "tol": tol, "track": track}

    def add_phase_shifter_section(self, sec_id, tol=0.001, track=True):
        if sec_id in self.phase_shifter_sections:
            print("WARNING: channel is already added")
            return
        if sec_id in self.sase_sections:
            self.phase_shifter_sections[sec_id] = {"id": sec_id, "tol": tol, "track": track}

    def add_undulator(self, sec_id, tol=0.001, track=True):
        if sec_id in self.sase_sections:
            print("WARNING: channel is already added")
            return
        if sec_id in self.sase_sections:
            self.undulators[sec_id] = {"id": sec_id, "tol": tol, "track": track}

    def add_channel(self, channel, tol=None, track=True):
        if channel in self.channels:
            print("WARNING: channel is already added")
            return
        self.channels.append(channel)
        self.channels_tol.append(tol)
        self.channels_track.append(track)

    def is_diff(self, snap1, snap2):
        diff = False
        for i, ch in enumerate(self.channels):
            print(ch, snap1[ch][0], snap2[ch][0], type(snap1[ch][0]))
            
            if (self.channels_tol[i] is not None) and self.channels_track[i] and np.abs(snap1[ch][0] - snap2[ch][0]) > self.channels_tol[i]:
                print("diff channels: ", ch, np.abs(snap1[ch][0] - snap2[ch][0]), self.channels_tol[i])
                diff = True
                
        for sec_id in self.orbit_sections:
            if self.orbit_sections[sec_id]["tol"] is not None and self.orbit_sections[sec_id]["track"]:
                for name in snap1.columns.values:

                    if "."+sec_id + ".X" in name:
                        #print("debug X: ", name, "."+sec_id + ".X" in name)
                        tol = self.orbit_sections[sec_id]["tol"]
                        if np.abs(snap1[name][0] - snap2[name][0])> tol:
                            print("diff orbit: ", name, np.abs(snap1[name][0] - snap2[name][0]), " tol = ", tol)
                            diff = True
                    if "."+sec_id + ".Y" in name:
                        #print("debug Y: ", name, "." + sec_id + ".X" in name)
                        tol = self.orbit_sections[sec_id]["tol"]
                        if np.abs(snap1[name][0] - snap2[name][0])> tol:
                            print("diff orbit: ", name, np.abs(snap1[name][0] - snap2[name][0]), " tol = ", tol)
                            diff = True

        for sec_id in self.magnet_sections:
            if self.magnet_sections[sec_id]["track"]:
                for name in snap1.columns.values:
                    if self.magnet_prefix is not None and any(x in name for x in self.magnet_prefix) and sec_id in name:
                        # print("debug X: ", name, "."+sec_id + ".X" in name)
                        tol = self.magnet_sections[sec_id]["tol"]
                        if np.abs(snap1[name][0] - snap2[name][0]) > tol:
                            print("diff magnet: ", name, np.abs(snap1[name][0] - snap2[name][0]),  " tol = ", tol)
                            diff = True

        for sec_id in self.phase_shifter_sections:
            if self.phase_shifter_sections[sec_id]["track"]:
                for name in snap1.columns.values:

                    if "BPS." in name and sec_id in name:
                        # print("debug X: ", name, "."+sec_id + ".X" in name)
                        tol = self.phase_shifter_sections[sec_id]["tol"]
                        if np.abs(snap1[name][0] - snap2[name][0]) > tol:
                            print("diff magnet: ", name, np.abs(snap1[name][0] - snap2[name][0]),  " tol = ", tol)
                            diff = True

        for sec_id in self.undulators:
            if self.undulators[sec_id]["track"]:
                for name in snap1.columns.values:

                    if ("U40." in name or "U68." in name) and sec_id in name:
                        # print("debug X: ", name, "."+sec_id + ".X" in name)
                        tol = self.phase_shifter_sections[sec_id]["tol"]
                        if np.abs(snap1[name][0] - snap2[name][0]) > tol:
                            print("diff magnet: ", name, np.abs(snap1[name][0] - snap2[name][0]),  " tol = ", tol)
                            diff = True

        return diff
        
        
class SnapshotDB:
    def __init__(self, filename="test.p"):
        self.df = pd.DataFrame()
        self.filename = filename
        self.orbit_sections = {}
        self.magnet_sections = {}
        self.channels = []
    
    def add(self, df):
        self.df = self.df.append(df)
    
    def save(self, filename=None):
        if filename is None:
            filename = self.filename
        self.df.to_pickle(filename)
    
    def load(self):
        df = pd.read_pickle(self.filename)
        self.df = df.reset_index()
        self.analyse()
        return self.df

    def analyse(self):

        self.orbit_sections = {}
        self.magnet_sections = {}
        self.channels = []

        for col in self.df.columns:
            if "/" not in col and col != "timestamp":

                if ".X" in col or ".Y" in col:
                    #print(col)
                    sec_id = col.split(".")[-2]
                    if sec_id not in self.orbit_sections:
                        self.orbit_sections[sec_id] = [col]
                    else:
                        self.orbit_sections[sec_id].append(col)
                else:
                    sec_id = col.split(".")[-1]
                    if sec_id not in self.magnet_sections:
                        self.magnet_sections[sec_id] = [col]
                    else:
                        self.magnet_sections[sec_id].append(col)
            else:
                self.channels.append(col)

    def beam_on_filter(self, beam_on=True):
        if "XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED" in self.df:
            return self.df[self.df["XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED"] == beam_on]
        else:
            print("there is no channel: XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED")
            return self.df

    def load_pickle_obj(self, filename):
        with open(filename, 'rb') as f:
            raw_image = pickle.load(f)
        return raw_image


    def get_background(self, parent_dir, cam_channel="XFEL.DIAG/CAMERA/OTRA.473.B2D/IMAGE_EXT_ZMQ"):
        df2 = self.beam_on_filter(beam_on=False)

        if cam_channel not in df2:
            print("there is no CAMERA in DataFrame")
            return None

        bg_imgs = []
        for i, path in enumerate(df2[cam_channel].tolist()):
            filename = str(parent_dir) + "/" + path[2:-3] + "pcl"

            bg_imgs.append(self.load_pickle_obj(filename))
        bg_img = np.mean(bg_imgs, axis=0)
        return bg_img

    def get_data_slice(self, x, y, beam_on=True, calc_moments=True, start_inx=None, stop_inx=None):
        if beam_on:
            df_filtered = self.beam_on_filter()
        else:
            df_filtered = self.df

        x_np = df_filtered[x].to_numpy()
        x_un = np.unique(x_np)[start_inx:stop_inx]
        data = {}

        for xi in x_un:
            y_raw = df_filtered[df_filtered[x] == xi][y].to_numpy()
            data[xi] = {"raw": y_raw}
            if calc_moments:
                data[xi]["mean"] = np.mean(y_raw)
                data[xi]["std"] = np.std(y_raw)
        return data

    def get_data(self, x, y, beam_on=True, calc_moments=True, start_inx=None, stop_inx=None):
        if y.__class__ is list:
            data_slices = []
            for yi in np.array(y):
                data = self.get_data_slice(x, yi, beam_on=beam_on, calc_moments=calc_moments, start_inx=start_inx, stop_inx=stop_inx)
                data_slices.append(data)
        else:
            data_slices = self.get_data_slice(x, y, beam_on=beam_on, calc_moments=calc_moments, start_inx=start_inx, stop_inx=stop_inx)
        return data_slices

    def plot(self, x, y, beam_on=True, ylabel="a.u.", start_inx=None, stop_inx=None):
        datas = self.get_data(x, y, beam_on=beam_on, start_inx=start_inx, stop_inx=stop_inx)

        fig, ax1 = plt.subplots()

        if datas.__class__ is not list:
            y = [y]
            datas = [datas]

        for i, data in enumerate(datas):
            x_un = np.array(list(data.keys()))# [start_inx:stop_inx]
            y_mean = np.array([data[xi]["mean"] for xi in x_un])
            y_std = np.array([data[xi]["std"] for xi in x_un])
            print(x_un.shape, y_mean.shape, y_std.shape)
            ax1.errorbar(x_un, y_mean, yerr=y_std, label=y[i])

        plt.xlabel(x)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

    def get_orbits(self, section_id):
        if self.orbit_sections is None:
            print("use .load() or .analyse() before")
            return
        if section_id in self.orbit_sections:
            x_bpm_names = [bpm for bpm in self.orbit_sections[section_id] if ".X" in bpm]
            y_bpm_names = [bpm for bpm in self.orbit_sections[section_id] if ".Y" in bpm]
            dfx = self.df[x_bpm_names].dropna(how="all", axis=1) # axis 1 means columns
            dfy = self.df[y_bpm_names].dropna(how="all", axis=1) # axis 1 means columns

            return dfx, dfy
        else:
            print("section_id is not in ", self.orbit_sections.keys())

    def get_orbit_dicts(self, section_id, legend_item=None):
        if self.orbit_sections is None:
            print("use .load() or .analyse() before")
            return

        if legend_item is None or not legend_item in self.df.columns:
            print("check legend")
            np_leg = np.arange(self.df.shape[0])  # vector with length of df row count
        else:
            np_leg = self.df[legend_item].to_numpy()
        if section_id not in self.orbit_sections:
            print("section id is not in self.orbit_sections")
            return None
        print("here")
        x_bpm_names = [bpm for bpm in self.orbit_sections[section_id] if ".X" in bpm]
        y_bpm_names = [bpm for bpm in self.orbit_sections[section_id] if ".Y" in bpm]
        
        x_orbits = {}
        y_orbits = {}
        for leg in np.unique(np_leg):
            if legend_item is None:
                df_tmp = self.df.iloc[leg].to_frame().T
            else:
                df_tmp = self.df[self.df[legend_item] == leg]

            dfx = df_tmp[x_bpm_names].dropna(how="all", axis=1)  # axis 1 means columns
            dfy = df_tmp[y_bpm_names].dropna(how="all", axis=1)  # axis 1 means columns
            if dfx.shape[1] ==0:
                continue
            dfx = dfx.dropna(how="any", axis=0)
            dfy = dfy.dropna(how="any", axis=0)
            x = dfx.to_numpy().mean(axis=0)
            y = dfy.to_numpy().mean(axis=0)
            x_orbits[leg] = x
            y_orbits[leg] = y

        return x_orbits, y_orbits, x_bpm_names, y_bpm_names


    def plot_orbit(self, section_id, legend_item=None, subtract_first=False, halve=False):

        x_orbits, y_orbits, x_bpm_names, y_bpm_names = self.get_orbit_dicts(section_id=section_id, legend_item=legend_item)
        fig = plt.figure(10, figsize=(9, 7))
        st = fig.suptitle("orbit", fontsize="x-large")
        ax_x= plt.subplot(211)
        for i, leg in enumerate(x_orbits):
            if halve and i%2 == 1:
                continue
            if subtract_first:
                ax_x.plot(x_orbits[leg] - x_orbits[list(x_orbits.keys())[0]], label=str(leg))
            else:
                ax_x.plot(x_orbits[leg], label=str(leg))

        plt.setp(ax_x.get_xticklabels(), visible=False)
        plt.ylabel("X [mm]")
        plt.legend()

        ax_y = plt.subplot(212, sharex=ax_x)
        for i, leg in enumerate(y_orbits):
            if halve and i%2 == 1:
                continue
            if subtract_first:
                ax_y.plot(y_orbits[leg] - y_orbits[list(y_orbits.keys())[0]], label=str(leg))
            else:
                ax_y.plot(y_orbits[leg], label=str(leg))
        bpm_names = [bpm.replace(".Y", "") for bpm in y_bpm_names]
        plt.xticks(np.arange(len(bpm_names)),
                   bpm_names)
        plt.xticks(rotation=90)
        plt.ylabel("Y [mm]")
        plt.legend()
        plt.show()
        
    def dict_append(self, dict1, dict2):
        dict3 = {}
        for key in dict2:
            data2 = dict2[key]
            if key in dict1:
                data1 = dict1[key]
            else:
                data1 = []
            dict3[key] = np.append(data1, data2)
        return dict3
    
    
    def plot_orbits(self, section_ids, legend_item=None, subtract_first=False, halve=False):
        print(section_ids)
        x_orbits_ids, y_orbits_ids, x_bpm_names_ids, y_bpm_names_ids = {}, {}, [], []
        for sec_id in section_ids:
            print(sec_id)
            x_orbits, y_orbits, x_bpm_names, y_bpm_names = self.get_orbit_dicts(section_id=sec_id, legend_item=legend_item)
            x_orbits_ids = self.dict_append(x_orbits_ids, x_orbits)
            y_orbits_ids = self.dict_append(y_orbits_ids, y_orbits)
                
            x_bpm_names_ids = np.append(x_bpm_names_ids, x_bpm_names)
            y_bpm_names_ids = np.append(y_bpm_names_ids, y_bpm_names)
            
            
        fig = plt.figure(10, figsize=(9, 7))
        st = fig.suptitle("orbit", fontsize="x-large")
        ax_x= plt.subplot(211)
        for i, leg in enumerate(x_orbits_ids):
            if halve and i%2 == 1:
                continue
            if subtract_first:
                ax_x.plot(x_orbits_ids[leg] - x_orbits_ids[list(x_orbits_ids.keys())[0]], label=str(leg))
            else:
                ax_x.plot(x_orbits_ids[leg], label=str(leg))

        plt.setp(ax_x.get_xticklabels(), visible=False)
        plt.ylabel("X [mm]")
        plt.legend()

        ax_y = plt.subplot(212, sharex=ax_x)
        for i, leg in enumerate(y_orbits_ids):
            if halve and i%2 == 1:
                continue
            if subtract_first:
                ax_y.plot(y_orbits_ids[leg] - y_orbits_ids[list(y_orbits_ids.keys())[0]], label=str(leg))
            else:
                ax_y.plot(y_orbits_ids[leg], label=str(leg))
        bpm_names = [bpm.replace(".Y", "") for bpm in y_bpm_names_ids]
        plt.xticks(np.arange(len(bpm_names)),
                   bpm_names)
        plt.xticks(rotation=90)
        plt.ylabel("Y [mm]")
        plt.legend()
        plt.show()
        
        

    def __str__(self):
        return self.df.__str__()