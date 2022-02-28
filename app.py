from collections import deque
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

import nils.reconstruction_module as recon
import spectralvd


class ReadThread(qtc.QThread):

    new_raw_reading = qtc.pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, float)
    new_clean_reading = qtc.pyqtSignal(np.ndarray, np.ndarray)
    new_rf_reading = qtc.pyqtSignal(np.ndarray)
    new_combined_reading = qtc.pyqtSignal(np.ndarray, np.ndarray, np.ndarray)

    crisp_channel = "XFEL.SDIAG/THZ_SPECTROMETER.FORMFACTOR/CRD.1934.TL/"

    def __init__(self, shots=10):
        super().__init__()

        self.shot_frequency = 10    # Hz

        self.rfs = deque(maxlen=shots)
        self.ffs = deque(maxlen=shots)
        self.charges = deque(maxlen=shots)
        self.nbunch = 0
        self.t_last = time.time()
        self.executor = ThreadPoolExecutor()

    def run(self):
        while True:
            self.wait_for_next_shot()

            rf, freqs, ff, ff_noise, detlim, charge = self.read()
            self.new_raw_reading.emit(freqs, ff, ff_noise, detlim, charge)

            freqs_clean, ff_clean, _ = recon.cleanup_formfactor(freqs, ff, ff_noise, detlim, channels_to_remove=[])
            self.new_clean_reading.emit(freqs_clean, ff_clean)

            self.new_rf_reading.emit(rf)
            self.new_combined_reading.emit(rf, freqs_clean, ff_clean)
    
    def wait_for_next_shot(self):
        shot_dt = 1 / self.shot_frequency

        t_passed = time.time() - self.t_last
        t_remaining = shot_dt - t_passed
        if t_remaining > 0:
            time.sleep(t_remaining)
        self.t_last = time.time()
    
    def read(self):
        rf_future = self.executor.submit(self.get_rf)
        charge_future = self.executor.submit(self.get_charge)
        ff_future = self.executor.submit(self.get_formfactor)
        detlim_future = self.executor.submit(self.get_detlim)
        
        rf = rf_future.result()
        charge = charge_future.result()
        freqs, ff_sq = ff_future.result()
        detlim = detlim_future.result()

        self.rfs.append(rf)
        self.ffs.append(ff_sq)
        self.charges.append(charge)

        rf_mean = np.array(self.rfs).mean(axis=0)

        ff_sq_mean = np.array(self.ffs).mean(axis=0)
        ff = np.sqrt(np.abs(ff_sq_mean)) * np.sign(ff_sq_mean)

        ff_sq_noise = np.std(np.array(self.ffs), axis=0)
        ff_noise = np.abs(0.5 / ff * ff_sq_noise)

        detlim_mean = np.sqrt(detlim * np.sqrt(10 / len(self.ffs)))
        
        charge_mean = np.mean(self.charges)

        return rf_mean, freqs, ff, ff_noise, detlim_mean, charge_mean
    
    def get_rf(self):
        facility = "XFEL.RF"
        device = "LLRF.CONTROLLER"
        locations = ["VS.A1.I1", "VS.AH1.I1", "VS.A2.L1", "VS.A3.L2"]
        properties = ["AMPL.SAMPLE", "PHASE.SAMPLE"]

        rf = [pydoocs.read(f"{facility}/{device}/{l}/{p}")["data"] for l in locations for p in properties]

        return rf

    def get_charge(self):
        charge = pydoocs.read(self.crisp_channel + "CHARGE.TD")["data"][0,1] * 1e-9
        return charge
    
    def get_formfactor(self):
        freqs = pydoocs.read(self.crisp_channel + "FORMFACTOR.XY")["data"][:,0] * 1e12
        ff_sq = pydoocs.read(self.crisp_channel + "FORMFACTOR.ARRAY")["data"][:,self.nbunch*2]
        return freqs, ff_sq

    def get_detlim(self):
        detlim = pydoocs.read(self.crisp_channel + "FORMFACTOR_MEAN_DETECTLIMIT.XY")["data"][:,1]
        return detlim
    
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

    def submit_reconstruction(self):
        raise NotImplementedError
    
    def set_active(self, active_state):
        if active_state:
            self._active_event.set()
        else:
            self._active_event.clear()
    
    def reconstruct(self):
        raise NotImplementedError


class NilsThread(ReconstructionThread):

    def submit_reconstruction(self, freqs, ff, ff_noise, detlim, charge):
        self.freqs, self.ff, self.ff_noise, self.detlim, self.charge = freqs, ff, ff_noise, detlim, charge
        self._new_crisp_reading_event.set()
    
    def reconstruct(self):
        t, current, _ = recon.master_recon(self.freqs, self.ff, self.ff_noise, self.detlim, self.charge,
                                           method="KKstart", channels_to_remove=[], show_plots=False)

        s = t * constants.speed_of_light

        return s, current


class LockmANNThread(ReconstructionThread):

    def __init__(self, path):
        super().__init__()

        self.model = spectralvd.LockmANN.load(path)

    def submit_reconstruction(self, freqs, ff, ff_noise, detlim, charge):
        self.freqs, self.ff, self.ff_noise, self.detlim, self.charge = freqs, ff, ff_noise, detlim, charge
        self._new_crisp_reading_event.set()
    
    def reconstruct(self):
        prediction = self.model.predict([(self.freqs, self.ff, self.ff_noise, self.detlim, self.charge)]*2)

        s = prediction[0][0]
        current = prediction[0][1]

        return s, current


class ANNRFThread(ReconstructionThread):

    def __init__(self, path):
        super().__init__()

        self.model = spectralvd.AdaptiveANNRF.load(path)

    def submit_reconstruction(self, rf):
        self.rf = rf
        self._new_crisp_reading_event.set()
    
    def reconstruct(self):
        prediction = self.model.predict([self.rf]*2)

        s = prediction[0][0]
        current = prediction[0][1]

        return s, current


class KNNTHzThread(ReconstructionThread):

    def __init__(self, path):
        super().__init__()

        self.model = spectralvd.AdaptiveKNNTHz.load(path)

    def submit_reconstruction(self, freqs, ff):
        self.freqs, self.ff = freqs, ff
        self._new_crisp_reading_event.set()
    
    def reconstruct(self):
        prediction = self.model.predict([(self.freqs, self.ff)]*2)

        s = prediction[0][0]
        current = prediction[0][1]

        return s, current


class ANNTHzThread(ReconstructionThread):

    def __init__(self, path):
        super().__init__()

        self.model = spectralvd.AdaptiveANNTHz.load(path)

    def submit_reconstruction(self, freqs, ff):
        self.freqs, self.ff = freqs, ff
        self._new_crisp_reading_event.set()
    
    def reconstruct(self):
        prediction = self.model.predict([(self.freqs, self.ff)]*2)

        s = prediction[0][0]
        current = prediction[0][1]

        return s, current


class ANNRFTHzThread(ReconstructionThread):

    def __init__(self, path):
        super().__init__()

        self.model = spectralvd.AdaptiveANNRFTHz.load(path)

    def submit_reconstruction(self, rf, freqs, ff):
        self.rf, self.freqs, self.ff = rf, freqs, ff
        self._new_crisp_reading_event.set()
    
    def reconstruct(self):
        prediction = self.model.predict([self.rf]*2, [(self.freqs,self.ff)]*2)

        s = prediction[0][0]
        current = prediction[0][1]

        return s, current


class FormfactorPlot(pg.PlotWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        pen = pg.mkPen("c", width=2)
        self.addLegend()
        self.plot_crisp = self.plot(range(999), np.ones(999), pen=pen, name="CRISP")
        # self.setXRange(int(684283010000), int(58267340000000))
        # self.setYRange(10e-3, 2)
        self.setLogMode(x=True, y=True)
        self.setLabel("bottom", text="Frequency", units="Hz")
        self.setLabel("left", text="|Frequency|")
        self.showGrid(x=True, y=True)

        self.t_last = time.time()
    
    def update(self, freqs, ff, ff_noise, detlim, charge):
        freqs_scaled = freqs.copy() # np.log10(frequency)
        ff_scaled = ff.copy()
        ff_scaled[ff_scaled <= 0] = 1e-3

        self.plot_crisp.setData(freqs_scaled, ff_scaled)


class CurrentPlot(pg.PlotWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        limit = 0.0001
        s = np.linspace(-limit, limit, 100)
        
        nils_pen = pg.mkPen(qtg.QColor(0, 128, 255), width=3)
        lockmann_pen = pg.mkPen(qtg.QColor(0, 0, 255), width=3)
        annrf_pen = pg.mkPen(qtg.QColor(255, 255, 0), width=3)
        annthz_pen = pg.mkPen(qtg.QColor(255, 0, 0), width=3)
        annrfthz_pen = pg.mkPen(qtg.QColor(255, 0, 255), width=3)
        knnthz_pen = pg.mkPen(qtg.QColor(0, 255, 0), width=3)
        
        self.addLegend()
        self.nils_plot = self.plot(s, np.zeros(100), pen=nils_pen, name="Nils")
        self.lockmann_plot = self.plot(s, np.zeros(100), pen=lockmann_pen, name="LockmANN")
        self.annrf_plot = self.plot(s, np.zeros(100), pen=annrf_pen, name="ANN RF")
        self.annthz_plot = self.plot(s, np.zeros(100), pen=annthz_pen, name="ANN THz")
        self.annrfthz_plot = self.plot(s, np.zeros(100), pen=annrfthz_pen, name="ANN RF+THz")
        self.knnthz_plot = self.plot(s, np.zeros(100), pen=knnthz_pen, name="KNN THz")
        self.setXRange(-limit, limit)
        self.setYRange(0, 10e3)
        self.setLabel("bottom", text="s", units="m")
        self.setLabel("left", text="Current", units="A")
        self.showGrid(x=True, y=True)

        self._nils_hidden = False
        self._lockmann_hidden = False
        self._annrf_hidden = False
        self._annthz_hidden = False
        self._annrfthz_hidden = False
        self._knnthz_hidden = False
    
    def update_nils(self, s, current):
        self.nils_s_scaled = s                # * 1e6
        self.nils_current_scaled = current    # * 1e-3

        if not self._nils_hidden:
            self.nils_plot.setData(self.nils_s_scaled, self.nils_current_scaled)
    
    def update_lockmann(self, s, current):
        self.lockmann_s_scaled = s                # * 1e6
        self.lockmann_current_scaled = current    # * 1e-3

        if not self._lockmann_hidden:
            self.lockmann_plot.setData(self.lockmann_s_scaled, self.lockmann_current_scaled)
    
    def update_annrf(self, s, current):
        self.annrf_s_scaled = s                # * 1e6
        self.annrf_current_scaled = current    # * 1e-3

        if not self._annrf_hidden:
            self.annrf_plot.setData(self.annrf_s_scaled, self.annrf_current_scaled)
    
    def update_annthz(self, s, current):
        self.annthz_s_scaled = s                # * 1e6
        self.annthz_current_scaled = current    # * 1e-3

        if not self._annthz_hidden:
            self.annthz_plot.setData(self.annthz_s_scaled, self.annthz_current_scaled)
    
    def update_annrfthz(self, s, current):
        self.annrfthz_s_scaled = s                # * 1e6
        self.annrfthz_current_scaled = current    # * 1e-3

        if not self._annrfthz_hidden:
            self.annrfthz_plot.setData(self.annrfthz_s_scaled, self.annrfthz_current_scaled)
    
    def update_knnthz(self, s, current):
        self.knnthz_s_scaled = s                # * 1e6
        self.knnthz_current_scaled = current    # * 1e-3

        if not self._knnthz_hidden:
            self.knnthz_plot.setData(self.knnthz_s_scaled, self.knnthz_current_scaled)
    
    def hide_nils(self, show):
        self._nils_hidden = not show
        if show:
            self.nils_plot.setData(self.nils_s_scaled, self.nils_current_scaled)
        else:
            # self.nils_plot.clear()
            self.nils_plot.setData([], [])
    
    def hide_lockmann(self, show):
        self._lockmann_hidden = not show
        if show:
            self.lockmann_plot.setData(self.lockmann_s_scaled, self.lockmann_current_scaled)
        else:
            # self.lockmann_plot.clear()
            self.lockmann_plot.setData([], [])
    
    def hide_annrf(self, show):
        self._annrf_hidden = not show
        if show:
            self.annrf_plot.setData(self.annrf_s_scaled, self.annrf_current_scaled)
        else:
            # self.annrf_plot.clear()
            self.annrf_plot.setData([], [])
    
    def hide_annthz(self, show):
        self._annthz_hidden = not show
        if show:
            self.annthz_plot.setData(self.annthz_s_scaled, self.annthz_current_scaled)
        else:
            # self.annthz_plot.clear()
            self.annthz_plot.setData([], [])
    
    def hide_annrfthz(self, show):
        self._annrfthz_hidden = not show
        if show:
            self.annrfthz_plot.setData(self.annrfthz_s_scaled, self.annrfthz_current_scaled)
        else:
            # self.annrfthz_plot.clear()
            self.annrfthz_plot.setData([], [])
    
    def hide_knnthz(self, show):
        self._knnthz_hidden = not show
        if show:
            self.knnthz_plot.setData(self.knnthz_s_scaled, self.knnthz_current_scaled)
        else:
            # self.knnthz_plot.clear()
            self.knnthz_plot.setData([], [])


class App(qtw.QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Spectral Virtual Diagnostics at European XFEL")

        self.nils_checkbox = qtw.QCheckBox("Nils")
        self.nils_checkbox.setChecked(True)
        self.lockmann_checkbox = qtw.QCheckBox("LockmANN")
        self.lockmann_checkbox.setChecked(True)
        self.annrf_checkbox = qtw.QCheckBox("ANN RF")
        self.annrf_checkbox.setChecked(True)
        self.annthz_checkbox = qtw.QCheckBox("ANN THz")
        self.annthz_checkbox.setChecked(True)
        self.annrfthz_checkbox = qtw.QCheckBox("ANN RF+THz")
        self.annrfthz_checkbox.setChecked(True)
        self.knnthz_checkbox = qtw.QCheckBox("KNN THz")
        self.knnthz_checkbox.setChecked(True)

        self.l1 = qtw.QLabel("N bunch: x2 ")
        self.sb_nbunch = qtw.QSpinBox()
        self.sb_nbunch.setMaximum(1024)

        self.formfactor_plot = FormfactorPlot()
        self.current_plot = CurrentPlot()

        self.read_thread = ReadThread()
        self.nils_thread = NilsThread()
        self.lockmann_thread = LockmANNThread("models/annthz")
        self.annrf_thread = ANNRFThread("models/annrf")
        self.annthz_thread = ANNTHzThread("models/annthz")
        self.annrfthz_thread = ANNRFTHzThread("models/annrfthz")
        self.knnthz_thread = KNNTHzThread("models/knnthz")

        self.read_thread.new_raw_reading.connect(self.formfactor_plot.update)
        self.read_thread.new_raw_reading.connect(self.nils_thread.submit_reconstruction)
        self.read_thread.new_raw_reading.connect(self.lockmann_thread.submit_reconstruction)
        self.read_thread.new_rf_reading.connect(self.annrf_thread.submit_reconstruction)
        self.read_thread.new_clean_reading.connect(self.annthz_thread.submit_reconstruction)
        self.read_thread.new_combined_reading.connect(self.annrfthz_thread.submit_reconstruction)
        self.read_thread.new_clean_reading.connect(self.knnthz_thread.submit_reconstruction)
        
        self.nils_thread.new_reconstruction.connect(self.current_plot.update_nils)
        self.lockmann_thread.new_reconstruction.connect(self.current_plot.update_lockmann)
        self.annrf_thread.new_reconstruction.connect(self.current_plot.update_annrf)
        self.annthz_thread.new_reconstruction.connect(self.current_plot.update_annthz)
        self.annrfthz_thread.new_reconstruction.connect(self.current_plot.update_annrfthz)
        self.knnthz_thread.new_reconstruction.connect(self.current_plot.update_knnthz)

        self.nils_checkbox.stateChanged.connect(self.nils_thread.set_active)
        self.nils_checkbox.stateChanged.connect(self.current_plot.hide_nils)
        self.lockmann_checkbox.stateChanged.connect(self.lockmann_thread.set_active)
        self.lockmann_checkbox.stateChanged.connect(self.current_plot.hide_lockmann)
        self.annrf_checkbox.stateChanged.connect(self.annrf_thread.set_active)
        self.annrf_checkbox.stateChanged.connect(self.current_plot.hide_annrf)
        self.annthz_checkbox.stateChanged.connect(self.annthz_thread.set_active)
        self.annthz_checkbox.stateChanged.connect(self.current_plot.hide_annthz)
        self.annrfthz_checkbox.stateChanged.connect(self.annrfthz_thread.set_active)
        self.annrfthz_checkbox.stateChanged.connect(self.current_plot.hide_annrfthz)
        self.knnthz_checkbox.stateChanged.connect(self.knnthz_thread.set_active)
        self.knnthz_checkbox.stateChanged.connect(self.current_plot.hide_knnthz)
        self.sb_nbunch.valueChanged.connect(self.read_thread.set_nbunch)

        grid = qtw.QGridLayout()
        grid.addWidget(self.formfactor_plot, 0, 0, 1, 6)
        grid.addWidget(self.current_plot, 0, 6, 1, 6)
        grid.addWidget(self.nils_checkbox, 1, 6, 1, 1)
        grid.addWidget(self.lockmann_checkbox, 1, 7, 1, 1)
        grid.addWidget(self.annrf_checkbox, 1, 8, 1, 1)
        grid.addWidget(self.annthz_checkbox, 1, 9, 1, 1)
        grid.addWidget(self.annrfthz_checkbox, 1, 10, 1, 1)
        grid.addWidget(self.knnthz_checkbox, 1, 11, 1, 1)
        grid.addWidget(self.l1, 1, 0, 1, 1)
        grid.addWidget(self.sb_nbunch, 1, 1, 1, 1)

        self.setLayout(grid)

        self.read_thread.start()
        self.nils_thread.start()
        self.lockmann_thread.start()
        self.annrf_thread.start()
        self.annthz_thread.start()
        self.annrfthz_thread.start()
        self.knnthz_thread.start()

        # Turn off some of the plots at app startup
        self.lockmann_checkbox.setChecked(False)
        self.annrf_checkbox.setChecked(False)
        self.annrfthz_checkbox.setChecked(False)
        self.knnthz_checkbox.setChecked(False)

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
