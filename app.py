from concurrent.futures import ThreadPoolExecutor
import pickle
import sys
from threading import Event

import numpy as np
import pydoocs
import PyQt5.QtCore as qtc
import PyQt5.QtGui as qtg
import PyQt5.QtWidgets as qtw
import pyqtgraph as pg
from scipy import constants

from nils.crisp_live_nils import get_charge, get_real_crisp_data
from nils.reconstruction_module import cleanup_formfactor, master_recon
from nils.simulate_spectrometer_signal import get_crisp_signal
import spectralvd


class CRISPThread(qtc.QThread):

    new_reading = qtc.pyqtSignal(str, np.ndarray, float)
    nbunch = 0
    
    def set_nbunch(self, n):
        self.nbunch = n
    
    def run(self):
        with ThreadPoolExecutor() as executor:
            while True:
                grating_future = executor.submit(CRISPThread.get_grating)
                charge_future = executor.submit(get_charge, shots=1)
                grating = grating_future.result()
                reading_future = executor.submit(get_real_crisp_data, 
                                                 shots=1, 
                                                 which_set="both", 
                                                 nbunch=self.nbunch)
                charge = charge_future.result()
                reading = reading_future.result()

                self.new_reading.emit(grating, reading, charge)
    
    def get_grating():
        response = pydoocs.read("XFEL.SDIAG/THZ_SPECTROMETER.GRATINGMOVER/CRD.1934.TL/STATUS.STR")
        grating_raw = response["data"]
        return grating_raw[:-5].lower()


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
            self._new_crisp_reading_event.clear()
            self.new_reconstruction.emit(s, current)

    def submit_reconstruction(self, grating, crisp_reading, charge):
        self._grating = grating
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

        t, current, _ = master_recon(frequency, formfactor, formfactor_noise, detlim, charge,
                                     method="KKstart", channels_to_remove=[], show_plots=False)

        s = t * constants.speed_of_light

        return s, current


class ANNThread(ReconstructionThread):

    def __init__(self, model_name):
        super().__init__()

        self.model = spectralvd.AdaptiveANNTHz.load("models/annthz")
    
    def reconstruct(self):
        frequency, formfactor, formfactor_noise, detlim = self._crisp_reading

        clean_frequency, clean_formfactor, _ = cleanup_formfactor(frequency, formfactor,
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
    
    def update(self, grating, reading, charge):
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
        
        ann_both_pen = pg.mkPen(qtg.QColor(255, 0, 0), width=3)
        
        ann_low_pen = pg.mkPen("g", width=3)

        nils_pen = pg.mkPen(qtg.QColor(0, 128, 255), width=3)
        self.ann_both_plot = self.plot(s, np.zeros(100), pen=ann_both_pen, name="ANN Both")
        self.ann_low_plot = self.plot(s, np.zeros(100), pen=ann_low_pen, name="ANN Low")
        self.nils_plot = self.plot(s, np.zeros(100), pen=nils_pen, name="Nils")

        self.setXRange(-limit, limit)
        self.setYRange(0, 10e3)
        self.setLabel("bottom", text="s", units="m")
        self.setLabel("left", text="Current", units="A")
        self.addLegend()
        self.showGrid(x=True, y=True)

        self._nils_hidden = False
        self._ann_both_hidden = False
        self._ann_low_hidden = False
    
    def update_ann_both(self, s, current):
        self.ann_both_s_scaled = s                # * 1e6
        self.ann_both_current_scaled = current    # * 1e-3

        if not self._ann_both_hidden:
            self.ann_both_plot.setData(self.ann_both_s_scaled, self.ann_both_current_scaled)
    
    def hide_ann_both(self, show):
        self._ann_both_hidden = not show
        if show:
            self.ann_both_plot.setData(self.ann_both_s_scaled, self.ann_both_current_scaled)
        else:
            # self.ann_both_plot.clear()
            self.ann_both_plot.setData([], [])

    def update_ann_low(self, s, current):
        self.ann_low_s_scaled = s                # * 1e6
        self.ann_low_current_scaled = current    # * 1e-3

        if not self._ann_low_hidden:
            self.ann_low_plot.setData(self.ann_low_s_scaled, self.ann_low_current_scaled)
    
    def hide_ann_low(self, show):
        self._ann_low_hidden = not show
        if show:
            self.ann_low_plot.setData(self.ann_low_s_scaled, self.ann_low_current_scaled)
        else:
            # self.ann_low_plot.clear()
            self.ann_low_plot.setData([], [])
    
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
        self.ann_both_checkbox = qtw.QCheckBox("ANN Both")
        self.ann_both_checkbox.setChecked(True)
        self.ann_low_checkbox = qtw.QCheckBox("ANN Low")
        self.ann_low_checkbox.setChecked(True)
        self.l1 = qtw.QLabel("N bunch: x2 ")
        self.sb_nbunch = qtw.QSpinBox()
        self.sb_nbunch.setMaximum(1024)
        

        self.formfactor_plot = FormfactorPlot()
        self.current_plot = CurrentPlot()

        self.crisp_thread = CRISPThread()
        self.nils_thread = NilsThread()
        self.ann_both_thread = ANNThread("both")
        self.ann_low_thread = ANNThread("low")

        self.crisp_thread.new_reading.connect(self.formfactor_plot.update)
        self.crisp_thread.new_reading.connect(self.nils_thread.submit_reconstruction)
        self.crisp_thread.new_reading.connect(self.ann_both_thread.submit_reconstruction)
        self.crisp_thread.new_reading.connect(self.ann_low_thread.submit_reconstruction)
        self.sb_nbunch.valueChanged.connect(self.crisp_thread.set_nbunch)
        
        self.nils_thread.new_reconstruction.connect(self.current_plot.update_nils)
        self.ann_both_thread.new_reconstruction.connect(self.current_plot.update_ann_both)
        self.ann_low_thread.new_reconstruction.connect(self.current_plot.update_ann_low)

        self.nils_checkbox.stateChanged.connect(self.nils_thread.set_active)
        self.nils_checkbox.stateChanged.connect(self.current_plot.hide_nils)
        self.ann_both_checkbox.stateChanged.connect(self.ann_both_thread.set_active)
        self.ann_both_checkbox.stateChanged.connect(self.current_plot.hide_ann_both)
        self.ann_low_checkbox.stateChanged.connect(self.ann_low_thread.set_active)
        self.ann_low_checkbox.stateChanged.connect(self.current_plot.hide_ann_low)

        grid = qtw.QGridLayout()
        grid.addWidget(self.formfactor_plot, 0, 0, 1, 3)
        grid.addWidget(self.current_plot, 0, 3, 1, 3)
        grid.addWidget(self.ann_both_checkbox, 1, 3, 1, 1)
        grid.addWidget(self.ann_low_checkbox, 1, 4, 1, 1)
        grid.addWidget(self.nils_checkbox, 1, 5, 1, 1)
        grid.addWidget(self.l1, 1, 0, 1, 1)
        grid.addWidget(self.sb_nbunch, 1, 1, 1, 1)

        self.setLayout(grid)

        self.crisp_thread.start()
        self.nils_thread.start()
        self.ann_both_thread.start()
        self.ann_low_thread.start()

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
    window.show()

    app.aboutToQuit.connect(window.handle_application_exit)

    sys.exit(app.exec_())
