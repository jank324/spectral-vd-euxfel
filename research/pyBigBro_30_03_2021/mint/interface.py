"""
S.Tomin, 2018
"""
import time
import numpy as np


class MachineInterface(object):
    def __init__(self, args):
        self.debug = False

    def get_value(self, channel):
        """
        Getter function for a given Machine.

        :param channel: (str) String of the devices name used
        :return: Data from the read on the Control System, variable data type depending on channel
        """
        raise NotImplementedError

    @staticmethod
    def add_args(subparser):
        """
        Method that will add the Machine interface specific arguments to the
        command line argument parser.

        :param subparser: (ArgumentParser)
        :return:
        """
        return

    def set_value(self, channel, val):
        """
        Method to set value to a channel

        :param channel: (str) String of the devices name used
        :param val: value
        :return: None
        """
        raise NotImplementedError

    def send_to_logbook(self, *args, **kwargs):
        """
        Send information to the electronic logbook.

        :param args:
            Values sent to the method without keywork
        :param kwargs:
            Dictionary with key value pairs representing all the metadata
            that is available for the entry.
        :return: bool
            True when the entry was successfully generated, False otherwise.
        """
        pass

    def device_factory(self, pv):
        """
        Create a device for the given PV using the proper Device Class.

        :param pv: (str) The process variable for which to create the device.
        :return: (Device) The device instance for the given PV.
        """
        return Device(eid=pv)

    def add_conversion(self, element):
        """
        Create methods in a device to translate physical units to hardware units

        :param element: in general ocelot element
        :return:
        """
        pass




class Device(object):
    def __init__(self, eid=None):
        self.eid = eid
        self.id = eid
        self.values = []
        self.times = []
        self.simplex_step = 0
        self.mi = None
        self.tol = 0.001
        self.timeout = 5  # seconds
        self.target = None
        self.low_limit = 0
        self.high_limit = 0
        self.phys2hw_factor = 1.
        self.hw2phys_factor = 1.

    def set_value(self, val):
        self.values.append(val)
        self.times.append(time.time())
        self.target = val
        self.mi.set_value(self.eid, val)

    def set_low_limit(self, val):
        self.low_limit = val

    def set_high_limit(self, val):
        self.high_limit = val

    def get_value(self):
        val = self.mi.get_value(self.eid)
        return val

    def trigger(self):
        pass

    def wait(self):
        if self.target is None:
            return

        start_time = time.time()
        while start_time + self.timeout <= time.time():
            if np.abs(self.get_value()-self.target) < self.tol:
                return
            time.sleep(0.05)

    def state(self):
        """
        Check if device is readable

        :return: state, True if readable and False if not
        """
        state = True
        try:
            self.get_value()
        except:
            state = False
        return state

    def clean(self):
        self.values = []
        self.times = []

    def check_limits(self, value):
        limits = self.get_limits()
        if value < limits[0] or value > limits[1]:
            print('limits exceeded for ', self.id, " - ", value, limits[0], value, limits[1])
            return True
        return False

    def get_limits(self):
        return [self.low_limit, self.high_limit]

    def phys2hw(self, phys_val):
        """
        Method to translate physical units to hardware units, e.g. angle [rad] to current [A]

        :param phys_val: physical unit
        :return: hardware unit
        """
        hw_val = phys_val*self.phys2hw_factor
        return hw_val

    def hw2phys(self, hw_val):
        """
        Method to translate hardware units to physical units, e.g. current [A] to angle [rad]

        :param hw_val: hardware unit
        :return: physical unit
        """
        phys_val = hw_val*self.hw2phys_factor
        return phys_val
