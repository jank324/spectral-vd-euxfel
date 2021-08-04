"""
Sergey Tomin, XFEL/DESY, 2017
"""
from mint.interface import Device
from PyQt5 import QtGui, QtCore
import numpy as np
import time
from threading import Thread, Event
import logging

logger = logging.getLogger(__name__)


class Corrector(Device):
    def __init__(self, eid=None, server="XFEL", subtrain="SA1"):
        super(Corrector, self).__init__(eid=eid)
        self.subtrain = subtrain
        self.server = server

    def set_value(self, val):
        #self.values.append(val)
        #self.times.append(time.time())
        ch = self.server + ".MAGNETS/MAGNET.ML/" + self.eid + "/KICK_MRAD.SP"
        self.mi.set_value(ch, val)

    def get_value(self):
        ch = self.server + ".MAGNETS/MAGNET.ML/" + self.eid + "/KICK_MRAD.SP"
        val = self.mi.get_value(ch)
        return val

    def get_limits(self):
        ch_min = self.server+ ".MAGNETS/MAGNET.ML/" + self.id + "/MIN_KICK"
        min_kick = self.mi.get_value(ch_min)
        ch_max = self.server + ".MAGNETS/MAGNET.ML/" + self.id + "/MAX_KICK"
        max_kick = self.mi.get_value(ch_max)
        return [min_kick*1000, max_kick*1000]
    
    def is_ok(self):
        ch = self.server+ ".MAGNETS/MAGNET.ML/" + self.id + "/COMBINED_STATUS"
        status = int(self.mi.get_value(ch))
        power_bit = '{0:08b}'.format(status)[-2]
        busy_bit = '{0:08b}'.format(status)[-4]
        
        if power_bit == "1" and busy_bit == "0":
            return True
        else:
            return False


class MITwiss(Device):
    def __init__(self, eid=None, server="XFEL", subtrain="SA1"):
        super(MITwiss, self).__init__(eid=eid)
        self.subtrain = "SA1"
        self.server = server

    def get_tws(self, section):
        ch_beta_x =  self.server + ".UTIL/BEAM_PARAMETER/" + section + "/PROJECTED_X.BETA." + self.subtrain
        ch_alpha_x = self.server + ".UTIL/BEAM_PARAMETER/" + section + "/PROJECTED_X.ALPHA." + self.subtrain
        ch_beta_y =  self.server + ".UTIL/BEAM_PARAMETER/" + section + "/PROJECTED_Y.BETA." + self.subtrain
        ch_alpha_y = self.server + ".UTIL/BEAM_PARAMETER/" + section + "/PROJECTED_Y.ALPHA." + self.subtrain
        #ch_energy =  "XFEL.UTIL/BEAM_PARAMETER/" + section + "/PROJECTED_X.ENERGY.SA1"
        tws_dict = {}
        tws_dict['beta_x'] = self.mi.get_value(ch_beta_x)
        tws_dict['beta_y'] = self.mi.get_value(ch_beta_y)
        tws_dict['alpha_x']  = self.mi.get_value(ch_alpha_x)
        tws_dict['alpha_y']  = self.mi.get_value(ch_alpha_y)
        return tws_dict

class ChargeDoocs(Device):
    def __init__(self, eid="XFEL.FEEDBACK/FT1.LONGITUDINAL/MONITOR1/TARGET", server="XFEL", subtrain="SA1"):
        super(ChargeDoocs, self).__init__(eid=eid)


class MPS(Device):
    def __init__(self, eid=None, server="XFEL", subtrain="SA1"):
        super(MPS, self).__init__(eid=eid)
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


class CavityA1(Device):
    def __init__(self, eid, server="XFEL", subtrain="SA1"):
        super(CavityA1, self).__init__(eid=eid)
        self.subtrain = subtrain
        self.server = server

    def set_value(self, val):
        ch = self.server + ".RF/LLRF.CONTROLLER/" + self.eid + "/SP.AMPL"
        self.mi.set_value(ch, val)
        logger.debug("CavityA1, ch: " + ch + " V = " + str(val))

    def get_value(self):
        ch = self.server + ".RF/LLRF.CONTROLLER/" + self.eid + "/SP.AMPL"
        val = self.mi.get_value(ch)
        return val




class BPMUI:
    def __init__(self, ui=None):
        self.tableWidget = None
        self.row = 0
        self.col = 0
        self.alarm = False

    def get_value(self):
        x = float(self.tableWidget.item(self.row, 1).text())
        y = float(self.tableWidget.item(self.row, 2).text())
        return (x, y)

    def set_value(self, val):
        x = val[0]
        y = val[1]
        x = np.round(x, 4)
        y = np.round(y, 4)
        self.tableWidget.item(self.row, 1).setText(str(x))
        self.tableWidget.item(self.row, 2).setText(str(y))
        self.check_values(val)

    def get_spin_values(self):
        x = self.tableWidget.cellWidget(self.row, 1).value()
        y = self.tableWidget.cellWidget(self.row, 2).value()
        return (x, y)

    def set_spin_values(self, val):
        x = np.round(val[0], 5)
        y = np.round(val[1], 5)
        self.tableWidget.cellWidget(self.row, 1).setValue(x)
        self.tableWidget.cellWidget(self.row, 2).setValue(y)


    def check_values(self, vals):
        if np.max(np.abs(vals)) > 15.:
            self.tableWidget.item(self.row, 1).setBackground(QtGui.QColor(255, 0, 0))  # red
            self.tableWidget.item(self.row, 2).setBackground(QtGui.QColor(255, 0, 0))  # red
            self.alarm = True
        elif vals[0] == 0 and vals[1] == 0:
            self.tableWidget.item(self.row, 1).setBackground(QtGui.QColor(255, 0, 0))  # red
            self.tableWidget.item(self.row, 2).setBackground(QtGui.QColor(255, 0, 0))  # red
            self.alarm = True
        else:
            self.tableWidget.item(self.row, 1).setBackground(QtGui.QColor(89, 89, 89))  # grey
            self.tableWidget.item(self.row, 2).setBackground(QtGui.QColor(89, 89, 89))  # grey
            self.alarm = False

    def set_init_value(self, val):
        self.tableWidget.item(self.row, 1).setText(str(val))

    def get_init_value(self):
        return float(self.tableWidget.item(self.row, 1).text())

    def uncheck(self):
        item = self.tableWidget.item(self.row, 3)
        item.setCheckState(False)

    def check(self):
        item = self.tableWidget.item(self.row, 3)
        item.setCheckState(QtCore.Qt.Checked)

    def state(self):
        item = self.tableWidget.item(self.row, 3)
        state = item.checkState()
        return state

    def set_hide(self, hide):
        #if hide:
        #    self.uncheck()
        #else:
        #    self.check()
        self.tableWidget.setRowHidden(self.row, hide)

    def is_touched(self, state):
        if state:
            self.tableWidget.item(self.row, 0).setForeground(QtGui.QColor(255, 101, 101))  # red
        else:
            self.tableWidget.item(self.row, 0).setForeground(QtGui.QColor(255, 255, 255))  # white

class BPM(Device):
    def __init__(self, eid, server="XFEL", subtrain="SA1"):
        super(BPM, self).__init__(eid=eid)
        self.subtrain = subtrain
        self.server = server
        self.bpm_server = "BPM"

    def get_pos(self):
        ch_x = self.server + ".DIAG/" + self.bpm_server + "/" + self.eid + "/X." + self.subtrain
        ch_y = self.server + ".DIAG/" + self.bpm_server + "/" + self.eid + "/Y." + self.subtrain
        x = self.mi.get_value(ch_x)
        y = self.mi.get_value(ch_y)
        return x, y

    def get_pos_frontend(self):
        bpm_server = "BPM"
        ch_x = self.server + ".DIAG/" + bpm_server + "/" + self.eid + "/X." + self.subtrain
        ch_y = self.server + ".DIAG/" + bpm_server + "/" + self.eid + "/Y." + self.subtrain
        x = self.mi.get_value(ch_x)
        y = self.mi.get_value(ch_y)
        return x, y



    def get_mean_pos(self):
        ch_x = self.server + ".DIAG/" + self.bpm_server + "/" + self.eid + "/X.TD"
        ch_y = self.server + ".DIAG/" + self.bpm_server + "/" + self.eid + "/Y.TD"
        x = np.mean(np.array(self.mi.get_value(ch_x))[:, 1])
        y = np.mean(np.array(self.mi.get_value(ch_y))[:, 1])

        return x, y

    def get_charge(self):
        x = self.mi.get_value(self.server + ".DIAG/CHARGE.ML/" + self.eid + "/CHARGE." + self.subtrain)
        return x

    def get_ref_pos(self):
        """
        Ref orbit only exists in ORBIT server not BPM
        channel ../POS.ALL.REF returns list [validation, x, y, z_pos, bpm_name]
        where validation = 0 - valid readings,
              x and y - ref beam position
              z_pos - position of BPM and

        :return:
        """
        ch = self.server + ".DIAG/ORBIT/" + self.eid + "/POS." + self.subtrain + ".REF"
        valid, x, y, z_pos, name = self.mi.get_value(ch)[0]
        return valid, x, y

    def get_gold_pos(self):
        """
        Ref orbit only exists in ORBIT server not BPM
        channel ../POS.ALL.REF returns list [validation, x, y, z_pos, bpm_name]
        where validation = 0 - valid readings,
              x and y - ref beam position
              z_pos - position of BPM and

        :return:
        """
        ch = self.server + ".DIAG/ORBIT/" + self.eid + "/POS." + self.subtrain + ".GOLD"
        valid, x, y, z_pos, name = self.mi.get_value(ch)[0]
        return valid, x, y

class DeviceUI:
    def __init__(self, ui=None):
        self.tableWidget = None
        self.row = 0
        self.col = 0
        self.alarm = False

    def get_value(self):
        return self.tableWidget.cellWidget(self.row, self.col).value()

    def set_value(self, val):
        self.tableWidget.cellWidget(self.row, self.col).setValue(val)

    def set_init_value(self, val):
        val = np.round(val, 4) # "{:1.4e}".format(val)
        self.tableWidget.item(self.row, 1).setText(str(val))

    def get_init_value(self):
        return float(self.tableWidget.item(self.row, 1).text())

    def uncheck(self):
        item = self.tableWidget.item(self.row, 3)
        item.setCheckState(False)

    def check(self):
        item = self.tableWidget.item(self.row, 3)
        item.setCheckState(QtCore.Qt.Checked)

    def state(self):
        item = self.tableWidget.item(self.row, 3)
        state = item.checkState()
        return state

    def check_values(self, val, lims, warn=False):
        if warn:
            self.tableWidget.item(self.row, 0).setBackground(QtGui.QColor(255, 255, 0))  # yellow
        else:
            #print("grey")
            self.tableWidget.item(self.row, 0).setBackground(QtGui.QColor(89, 89, 89))  # grey
        self.alarm = False
        if not(lims[0] <= val <= lims[1]):
            self.tableWidget.item(self.row, 0).setBackground(QtGui.QColor(255, 0, 0))  # red
            self.alarm = True
    
    def set_fault(self, fault):
        if fault:
            self.tableWidget.item(self.row, 0).setBackground(QtGui.QColor(255, 255, 0)) # yellow
            self.tableWidget.item(self.row, 1).setBackground(QtGui.QColor(255, 255, 0)) # yellow
            self.tableWidget.item(self.row, 3).setBackground(QtGui.QColor(255, 255, 0)) # yellow
        else:
            self.tableWidget.item(self.row, 0).setBackground(QtGui.QColor(89, 89, 89)) # grey
            self.tableWidget.item(self.row, 1).setBackground(QtGui.QColor(89, 89, 89)) # grey
            self.tableWidget.item(self.row, 3).setBackground(QtGui.QColor(89, 89, 89)) # grey
            
            
    def check_diff(self, tol=0.01):
        ival = self.get_init_value()
        val = self.get_value()
        diff = np.abs(val - ival)
        if diff > tol:
            self.tableWidget.item(self.row, 1).setForeground(QtGui.QColor(255, 101, 101)) # red
        else:
            self.tableWidget.item(self.row, 1).setForeground(QtGui.QColor(255, 255, 255)) # white
    
    def set_hide(self, hide):
        #if hide and uncheck:
        #    self.uncheck()
        #else:
        #    self.check()
        self.tableWidget.setRowHidden(self.row, hide)

class MICavity(Device):
    def __init__(self, eid=None, server="XFEL", subtrain="SA1"):
        super(MICavity, self).__init__(eid=eid)
        self.subtrain = subtrain
        self.server = server

    def get_value(self):
        #C.A3.1.1.L2
        #M4.A4.L2
        parts = self.eid.split(".")
        eid = "M"+parts[2]+"."+parts[1]+"."+parts[4]
        ch = self.server + ".RF/LLRF.ENERGYGAIN.ML/" + eid + "/ENERGYGAIN.1" #+ self.subtrain
        val = self.mi.get_value(ch)/8.
        return val
    
    def get_phase(self):
        # XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.PHASE
        parts = self.eid.split(".")
        eid = "CTRL."+parts[1]+"."+parts[4]
        ch = self.server + ".RF/LLRF.CONTROLLER/" + eid + "/SP.PHASE" #+ self.subtrain
        phi = self.mi.get_value(ch)
        return phi
        

class MIOrbit(Device, Thread):
    def __init__(self, eid=None, server="XFEL", subtrain="SA1"):
        Device.__init__(self, eid=eid)
        Thread.__init__(self)
        #super(MIOrbit, self).__init__(eid=eid)
        self.subtrain = subtrain
        self.server = server
        self.bpm_server = "ORBIT"     # or "BPM"
        self.time_delay = 0.1         # sec
        self.charge_threshold = 0.005 # nC
        self.subtrain = subtrain
        self.bpm_names = []
        self.x = []
        self.y = []
        self.mean_x = []
        self.mena_y = []
        self.mean_charge = []
        #self.charge = []
    
    def run(self):
        start = time.time()
        print("RUN")
        #self.read_positions()
        self.mi.get_value(self.server + ".DIAG/" + self.bpm_server + "/*/X." + self.subtrain)
        print("RUN FINISH in ", time.time() - start, "sec")

    def read_positions(self, reliable_reading=False, suffix=""):

        if reliable_reading:
            nreadings = 30
            time_delay = 0.05
        else:
            nreadings = 1
            time_delay = 0
        try:
            for i in range(nreadings):
                orbit_x = self.mi.get_value(self.server + ".DIAG/" + self.bpm_server + "/*/X." + self.subtrain + suffix)
                orbit_y = self.mi.get_value(self.server + ".DIAG/" + self.bpm_server + "/*/Y." + self.subtrain + suffix)
                time.sleep(time_delay)
                #print(orbit_x)
                if orbit_x[0][1] != 0 and orbit_y[0][1] != 0:
                #if not(np.isnan(orbit_x[0]["float1"])) and not(np.isnan(orbit_y[0]["float1"])):
                    print("OK")
                    break
            #print(self.server + ".DIAG/" + self.bpm_server + "/*/X." + self.subtrain, orbit_x[0])
        except Exception as e:
            logger.critical("read_positions: self.mi.get_value: " + str(e))
            raise e
        #    print("ERROR: reading from DOOCS")
        #    return False
        #print(orbit_x)
        try:
            names_x = [data[4] for data in orbit_x]
            names_y = [data[4] for data in orbit_y]
        except Exception as e:
            logger.critical("read_positions: names_x = [data['str'] for data in orbit_x]" + str(e))
            raise e

        if not np.array_equal(names_x, names_y):
            logger.warning(" MIOrbit: read_positions: X and Y orbits are not equal")
        self.x = np.array([data[1] for data in orbit_x])
        self.y = np.array([data[1] for data in orbit_y])
        return [names_x, self.x, self.y]

    def read_charge(self, suffix=""):
        try:
            charge = self.mi.get_value(self.server + ".DIAG/CHARGE.ML/*/CHARGE." + self.subtrain + suffix)
        except Exception as e:
            logger.critical("read_charge: self.mi.get_value: " + str(e))
            raise e
        names = [data[4] for data in charge]
        values = np.array([data[1] for data in charge])
        return names, values

    def read_orbit(self, reliable_reading, suffix=""):
        names_xy, x, y = self.read_positions(reliable_reading, suffix=suffix)
        names_charge, charge = self.read_charge(suffix=suffix)
        indx = [not ("TORA." in name or "TORC." in name) for name in names_charge]
        names_charge = np.array(names_charge)[indx]
        charge = np.array(charge)[indx]
        
        #print( len(names_charge), len(names_xy))
        #for n_ch, n_xy in zip(names_charge, names_xy):
        #    if n_ch != n_xy:
        #       print(n_ch, n_xy)

        if not np.array_equal(names_xy, names_charge):
            logger.warning(" MIOrbit: read_orbit: CHARGE reading and POSITIONS are not equal")
            #return False
        return names_xy, x, y, charge


    def read_and_average(self, nreadings, take_last_n, reliable_reading=False, suffix=""):
        logger.info(" MIorbit: read_and_average")
        orbits_x = []
        orbits_y = []
        orbits_charge = []
        saved_names = []
        for i in range(nreadings):
            names, x, y, charge = self.read_orbit(reliable_reading=reliable_reading, suffix=suffix)
            orbits_x.append(x)
            orbits_y.append(y)
            orbits_charge.append(charge)
            if i > 0:
                if not np.array_equal(saved_names, names):
                    logger.warning(" MIOrbit: read_and_average: error: arrays are different ")
            saved_names = names
            time.sleep(self.time_delay)
        self.bpm_names = saved_names
        self.mean_x = np.mean(orbits_x[-take_last_n:], axis=0)
        self.mean_y = np.mean(orbits_y[-take_last_n:], axis=0)
        self.mean_charge = np.mean(orbits_charge[-take_last_n:], axis=0)
        return self.bpm_names, self.mean_x, self.mean_y, self.mean_charge

    def get_bpms(self, bpms):
        """
        All bpm works with [m] but doocs gives position in [mm]

        :param bpms: list of BPM objects
        :param charge_threshold:
        :return:
        """
        if len(self.bpm_names) == 0:
            return False
        #bpm_names = [bpm.id for bpm in bpms]
        indxs = []
        valid_bpm_inx = []
        for i, bpm in enumerate(bpms):
            if bpm.id not in self.bpm_names:
                bpm.ui.uncheck()
            else:
                valid_bpm_inx.append(i)
                indxs.append(self.bpm_names.index(bpm.id))
                logger.debug(" MIOrbit: get_bpms: len(bpm)="+ str(len(bpms)) + "  len(indxs) = " + str(len(indxs)))
        bpms = [bpms[indx] for indx in valid_bpm_inx]
        for i, bpm in enumerate(bpms):
            inx = indxs[i]
            bpm.x = self.mean_x[inx]/1000      # [mm] -> [m]
            bpm.y = self.mean_y[inx]/1000      # [mm] -> [m]
            bpm.charge = self.mean_charge[inx] # nC
        return True

    def read_doocs_ref_orbit(self):
        """
        Ref orbit only exists in ORBIT server not BPM
        channel ../POS.ALL.REF returns list [validation, x, y, z_pos, bpm_name]
        where validation = 0 - valid readings,
              x and y - ref beam position
              z_pos - position of BPM and

        :return:
        """
        ref_orbit = self.mi.get_value(self.server + ".DIAG/" + "ORBIT" + "/*/POS." + self.subtrain + ".REF")
        return ref_orbit

    def read_doocs_gold_orbit(self):
        """
        Golden orbit only exists in ORBIT server not BPM
        channel ../POS.ALL.REF returns list [validation, x, y, z_pos, bpm_name]
        where validation = 0 - valid readings,
              x and y - ref beam position
              z_pos - position of BPM and

        :return:
        """
        gold_orbit = self.mi.get_value(self.server + ".DIAG/" + "ORBIT" + "/*/POS." + self.subtrain + ".GOLD")
        return gold_orbit

class MIAdviser(Device):
    def __init__(self, eid=None, server="XFEL", subtrain="SA1"):
        super(MIAdviser, self).__init__(eid=eid)
        self.bpm_server = "BPM"  # "ORBIT"     # or "BPM"
        self.subtrain = subtrain
        self.server = server

    def get_x(self):
        try:
            self.orbit_x = self.mi.get_value(self.server + ".DIAG/" + self.bpm_server + "/*/X." + self.subtrain)
        except Exception as e:
            logger.info("get_x: self.mi.get_value: " + str(e))
            self.orbit_x = []

    def get_y(self):
        try:
            self.orbit_y = self.mi.get_value(self.server + ".DIAG/" + self.bpm_server + "/*/Y." + self.subtrain)
        except Exception as e:
            logger.info("get_y: self.mi.get_value: " + str(e))
            self.orbit_y = []
        
    def get_bpm_z_pos(self):
        try:
            self.bpm_z_pos = self.mi.get_value(self.server + ".DIAG/" + self.bpm_server + "/*/Z_POS")
        except Exception as e:
            logger.info("get_bpm_z_pos: self.mi.get_value: " + str(e))
            self.bpm_z_pos = []
        #print(self.bpm_z_pos)

    def get_kicks(self):
        #"XFEL.MAGNETS/MAGNET.ML/" + self.eid + "/KICK_MRAD.SP"
        try:
            self.kicks = self.mi.get_value(self.server + ".MAGNETS/MAGNET.ML/*/KICK_MRAD.SP")
        except Exception as e:
            logger.critical("get_kicks: self.mi.get_value: " + str(e))
            raise e

    def get_momentums(self):
        #"XFEL.MAGNETS/MAGNET.ML/" + self.eid + "/KICK_MRAD.SP"
        try:
            self.moments = self.mi.get_value(self.server + ".MAGNETS/MAGNET.ML/*/MOMENTUM.SP")
        except Exception as e:
            logger.critical("get_momentums: self.mi.get_value: " + str(e))
            raise e

    def get_cor_z_pos(self):
        try:
            self.cor_z_pos = self.mi.get_value("XFEL.MAGNETS/MAGNET.ML/*/Z_POS")
        except Exception as e:
            logger.info("get_cor_z_pos: self.mi.get_value: " + str(e))
            self.cor_z_pos = []

    def get_corrs(self, ref_names):

        self.get_kicks()
        self.get_momentums()
        self.get_cor_z_pos()
        names = [x["str"] for x in self.kicks]
        #print(self.kicks)
        kicks = np.array([x["float1"] for x in self.kicks])/1000.
        moments = np.array([x["float1"] for x in self.moments])
        z_poss = np.array([x["float1"] for x in self.cor_z_pos])
        indxs = []
        for name in ref_names:
            indxs.append(names.index(name))
        return kicks[indxs], moments[indxs], z_poss[indxs]

    def get_bpm_z_from_ref(self, ref_names):
        self.get_bpm_z_pos()
        names = [x["str"] for x in self.bpm_z_pos]
        z_poss = np.array([x["float1"] for x in self.bpm_z_pos])
        indxs = []
        for name in ref_names:
            indxs.append(names.index(name))
        
        return z_poss[indxs]
        
    def get_bpm_x(self, ref_names):

        self.get_x()
        if len(self.orbit_x) == 0:
            return None
        names = [x["str"] for x in self.orbit_x]
        pos = np.array([x["float1"] for x in self.orbit_x])

        indxs = []
        for name in ref_names:
            if name in names:
                indxs.append(names.index(name))
        
        z_pos = self.get_bpm_z_from_ref(ref_names)
        return pos[indxs], z_pos

    def get_bpm_y(self, ref_names):

        self.get_y()
        #self.get_bpm_z_pos()

        if len(self.orbit_y) == 0:
            return None

        names = [x["str"] for x in self.orbit_y]
        pos = np.array([x["float1"] for x in self.orbit_y])
        #z_poss = np.array([x["float"] for x in self.bpm_z_pos])

        indxs = []
        for name in ref_names:
            indxs.append(names.index(name))
            
        z_pos = self.get_bpm_z_from_ref(ref_names)
        return pos[indxs], z_pos


class MIStandardFeedback(Device):
    def __init__(self, eid=None, server="XFEL", subtrain="SA1"):
        super(MIStandardFeedback, self).__init__(eid=eid)
        self.subtrain = subtrain
        self.server = server

    def is_running(self):
        status = self.mi.get_value(self.server + ".FEEDBACK/ORBIT.SA1/ORBITFEEDBACK/ACTIVATE_FB")
        return status


class MISASE3Feedback(Device):
    def __init__(self, eid=None, server="XFEL", subtrain="SA1"):
        super(MISASE3Feedback, self).__init__(eid=eid)
        self.subtrain = subtrain
        self.server = server

    def is_running(self):
        status = self.mi.get_value(self.server + ".FEEDBACK/ORBIT.SA3/ORBITFEEDBACK/ACTIVATE_FB")
        return status


class MISASE2Feedback(Device):
    def __init__(self, eid=None, server="XFEL", subtrain="SA1"):
        super(MISASE2Feedback, self).__init__(eid=eid)
        self.subtrain = subtrain
        self.server = server

    def is_running(self):
        status = self.mi.get_value(self.server + ".FEEDBACK/ORBIT.SA2/ORBITFEEDBACK/ACTIVATE_FB")
        return status
        
