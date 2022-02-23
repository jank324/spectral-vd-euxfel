from concurrent.futures import ThreadPoolExecutor
import sys
from threading import Event
import time

import numpy as np
import pydoocs
import PyQt5.QtCore as qtc
import PyQt5.QtGui as qtg
import PyQt5.QtWidgets as qtw
import pyqtgraph as pg
from scipy import constants

import nils.crisp_live_nils as cl
import nils.reconstruction_module as rm
import spectralvd


class ReadThread(qtc.QThread):

    new_reading = qtc.pyqtSignal(np.ndarray, float)

    def __init__(self):
        super().__init__()

        self.nbunch = 0
        self.t_last = time.time()

    def run(self):
        with ThreadPoolExecutor() as executor:
            while True:
                t_passed = time.time() - self.t_last
                t_remaining = 0.1 - t_passed
                if t_remaining > 0:
                    time.sleep(t_remaining)
                t_now = time.time()
                dt = t_now - self.t_last
                self.t_last = t_now
                # print(f"Read thread running at {1/dt:.2f} Hz")

                charge_future = executor.submit(cl.get_charge, shots=1)
                reading_future = executor.submit(cl.get_real_crisp_data, 
                                                 shots=1, 
                                                 which_set="both", 
                                                 nbunch=self.nbunch)
                
                charge = charge_future.result()
                reading = reading_future.result()

                self.new_reading.emit(reading, charge)
    
    def set_nbunch(self, nbunch):
        self.nbunch = nbunch


class ReconstructionThread(qtc.QThread):

    new_reconstruction = qtc.pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self):
        super().__init__()
        self._new_crisp_reading_event = Event()
        self._new_crisp_reading_event.clear()

        self._active_event = Event()
        self._active_event.set()

    def run(self):
        while True:
            self._active_event.wait()
            self._new_crisp_reading_event.wait()

            s, current = self.reconstruct()
            self.new_reconstruction.emit(s, current)
            
            self._new_crisp_reading_event.clear()

    def submit_reconstruction(self, crisp_reading, charge):
        self._crisp_reading = crisp_reading
        self._charge = charge

        self._new_crisp_reading_event.set()
    
    def set_active(self, active_state):
        if active_state:
            self._active_event.set()
        else:
            self._active_event.clear()
    
    def reconstruct(self):
        raise NotImplementedError


class NilsThread(ReconstructionThread):
    
    def reconstruct(self):
        frequency, formfactor, formfactor_noise, detlim = self._crisp_reading
        charge = self._charge

        t, current, _ = rm.master_recon(frequency, formfactor, formfactor_noise, detlim, charge,
                                        method="KKstart", channels_to_remove=[], show_plots=False)

        s = t * constants.speed_of_light

        return s, current


class ANNTHzThread(ReconstructionThread):

    def __init__(self, path):
        super().__init__()

        self.model = spectralvd.AdaptiveANNTHz.load(path)
    
    def reconstruct(self):
        frequency, formfactor, formfactor_noise, detlim = self._crisp_reading

        clean_frequency, clean_formfactor, _ = rm.cleanup_formfactor(frequency, formfactor,
                                                                     formfactor_noise, detlim,
                                                                     channels_to_remove=[])

        prediction = self.model.predict([(clean_frequency, clean_formfactor)]*2)

        s = prediction[0][0]
        current = prediction[0][1]

        return s, current


class FormfactorPlot(pg.PlotWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        pen = pg.mkPen("c", width=2)
        self.plot_crisp = self.plot(range(999), np.ones(999), pen=pen, name="CRISP")
        # self.setXRange(int(684283010000), int(58267340000000))
        # self.setYRange(10e-3, 2)
        self.setLogMode(x=True, y=True)
        self.setLabel("bottom", text="Frequency", units="Hz")
        self.setLabel("left", text="|Frequency|")
        self.addLegend()
        self.showGrid(x=True, y=True)

        self.t_last = time.time()
    
    def update(self, reading, charge):
        t_now = time.time()
        dt = t_now - self.t_last
        self.t_last = t_now
        # print(f"Forffactor update running at {1/dt:.2f} Hz")

        frequency, formfactor, _, _ = reading

        frequency_scaled = frequency.copy() # np.log10(frequency)
        formfactor_scaled = formfactor.copy()
        formfactor_scaled[formfactor_scaled <= 0] = 1e-3
        #formfactor_scaled = formfactor_scaled + 1e-12
        #formfactor_scaled = np.log10(formfactor_scaled + 1)

        self.plot_crisp.setData(frequency_scaled, formfactor_scaled)


class CurrentPlot(pg.PlotWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        limit = 0.00020095917745111108 # * 1e6
        s = np.linspace(-limit, limit, 100)
        
        annthz_pen = pg.mkPen(qtg.QColor(255, 0, 0), width=3)
        
        nils_pen = pg.mkPen(qtg.QColor(0, 128, 255), width=3)
        self.annthz_plot = self.plot(s, np.zeros(100), pen=annthz_pen, name="ANN THz")
        self.nils_plot = self.plot(s, np.zeros(100), pen=nils_pen, name="Nils")

        self.setXRange(-limit, limit)
        self.setYRange(0, 10e3)
        self.setLabel("bottom", text="s", units="m")
        self.setLabel("left", text="Current", units="A")
        self.addLegend()
        self.showGrid(x=True, y=True)

        self._nils_hidden = False
        self._annthz_hidden = False

        self.t_last = time.time()
    
    def update_annthz(self, s, current):
        t_now = time.time()
        dt = t_now - self.t_last
        self.t_last = t_now
        # print(f"ANN update running at {1/dt:.2f} Hz")

        self.annthz_s_scaled = s                # * 1e6
        self.annthz_current_scaled = current    # * 1e-3

        if not self._annthz_hidden:
            self.annthz_plot.setData(self.annthz_s_scaled, self.annthz_current_scaled)
    
    def hide_annthz(self, show):
        self._annthz_hidden = not show
        if show:
            self.annthz_plot.setData(self.annthz_s_scaled, self.annthz_current_scaled)
        else:
            # self.annthz_plot.clear()
            self.annthz_plot.setData([], [])
    
    def update_nils(self, s, current):
        self.nils_s_scaled = s                # * 1e6
        self.nils_current_scaled = current    # * 1e-3

        if not self._nils_hidden:
            self.nils_plot.setData(self.nils_s_scaled, self.nils_current_scaled)
    
    def hide_nils(self, show):
        self._nils_hidden = not show
        if show:
            self.nils_plot.setData(self.nils_s_scaled, self.nils_current_scaled)
        else:
            # self.nils_plot.clear()
            self.nils_plot.setData([], [])


class App(qtw.QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Spectral Virtual Diagnostics at European XFEL")

        self.nils_checkbox = qtw.QCheckBox("Nils")
        self.nils_checkbox.setChecked(True)
        self.annthz_checkbox = qtw.QCheckBox("ANN THz")
        self.annthz_checkbox.setChecked(True)
        self.l1 = qtw.QLabel("N bunch: x2 ")
        self.sb_nbunch = qtw.QSpinBox()
        self.sb_nbunch.setMaximum(1024)

        self.formfactor_plot = FormfactorPlot()
        self.current_plot = CurrentPlot()

        self.read_thread = ReadThread()
        self.nils_thread = NilsThread()
        self.annthz_thread = ANNTHzThread("models/annthz")

        self.read_thread.new_reading.connect(self.formfactor_plot.update)
        self.read_thread.new_reading.connect(self.nils_thread.submit_reconstruction)
        self.read_thread.new_reading.connect(self.annthz_thread.submit_reconstruction)
        
        self.nils_thread.new_reconstruction.connect(self.current_plot.update_nils)
        self.annthz_thread.new_reconstruction.connect(self.current_plot.update_annthz)

        self.nils_checkbox.stateChanged.connect(self.nils_thread.set_active)
        self.nils_checkbox.stateChanged.connect(self.current_plot.hide_nils)
        self.annthz_checkbox.stateChanged.connect(self.annthz_thread.set_active)
        self.annthz_checkbox.stateChanged.connect(self.current_plot.hide_annthz)
        self.sb_nbunch.valueChanged.connect(self.read_thread.set_nbunch)

        grid = qtw.QGridLayout()
        grid.addWidget(self.formfactor_plot, 0, 0, 1, 3)
        grid.addWidget(self.current_plot, 0, 3, 1, 3)
        grid.addWidget(self.annthz_checkbox, 1, 3, 1, 1)
        grid.addWidget(self.nils_checkbox, 1, 4, 1, 1)
        grid.addWidget(self.l1, 1, 0, 1, 1)
        grid.addWidget(self.sb_nbunch, 1, 1, 1, 1)

        self.setLayout(grid)

        self.read_thread.start()
        self.nils_thread.start()
        self.annthz_thread.start()

    def handle_application_exit(self):
        pass
    

if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)

    # Force the style to be the same on all OSs
    app.setStyle("Fusion")

    # Now use a palette to switch to dark colors
    palette = qtg.QPalette()
    palette.setColor(qtg.QPalette.Window, qtg.QColor(53, 53, 53))
    palette.setColor(qtg.QPalette.WindowText, qtc.Qt.white)
    palette.setColor(qtg.QPalette.Base, qtg.QColor(25, 25, 25))
    palette.setColor(qtg.QPalette.AlternateBase, qtg.QColor(53, 53, 53))
    palette.setColor(qtg.QPalette.ToolTipBase, qtc.Qt.white)
    palette.setColor(qtg.QPalette.ToolTipText, qtc.Qt.white)
    palette.setColor(qtg.QPalette.Text, qtc.Qt.white)
    palette.setColor(qtg.QPalette.Button, qtg.QColor(53, 53, 53))
    palette.setColor(qtg.QPalette.ButtonText, qtc.Qt.white)
    palette.setColor(qtg.QPalette.BrightText, qtc.Qt.red)
    palette.setColor(qtg.QPalette.Link, qtg.QColor(42, 130, 218))
    palette.setColor(qtg.QPalette.Highlight, qtg.QColor(42, 130, 218))
    palette.setColor(qtg.QPalette.HighlightedText, qtc.Qt.black)
    app.setPalette(palette)

    window = App()
    window.resize(1200, 600)
    window.show()

    app.aboutToQuit.connect(window.handle_application_exit)

    sys.exit(app.exec_())
