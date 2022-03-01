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

import nils.crisp_functions as crisp
import nils.reconstruction_module as recon
import nils.reconstruction_module_after_diss as adiss
import spectralvd


COLORS = [
    (138, 176, 207),
    (243, 182, 112),
    (188, 219, 120),
    (231, 133, 119),
    (179, 134, 185),
    (253, 236, 130)
]


class ReadThread(qtc.QThread):

    new_raw_reading = qtc.pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, float)
    new_clean_reading = qtc.pyqtSignal(np.ndarray, np.ndarray)
    new_rf_reading = qtc.pyqtSignal(np.ndarray)
    new_combined_reading = qtc.pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    new_stage_reading = qtc.pyqtSignal(str)

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

            rf, freqs, ff, ff_noise, detlim, charge, stage = self.read()
            self.new_raw_reading.emit(freqs, ff, ff_noise, detlim, charge)

            freqs_clean, ff_clean, _ = recon.cleanup_formfactor(freqs, ff, ff_noise, detlim, channels_to_remove=[])
            self.new_clean_reading.emit(freqs_clean, ff_clean)

            self.new_rf_reading.emit(rf)
            self.new_combined_reading.emit(rf, freqs_clean, ff_clean)

            self.new_stage_reading.emit(stage)
    
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

        stage = crisp.get_stage_position()

        return rf_mean, freqs, ff, ff_noise, detlim_mean, charge_mean, stage
    
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
        reconstructed = adiss.master_recon(self.freqs, self.ff, self.ff_noise, self.detlim, self.charge,
                                           method="KKstart", channels_to_remove=[], show_plots=False)
        t, current = reconstructed[:2]

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


class ReverseThread(qtc.QThread):

    new_reversal = qtc.pyqtSignal(np.ndarray)
    active_state_changed = qtc.pyqtSignal(bool)

    def __init__(self, path):
        super().__init__()
        self._new_crisp_reading_event = Event()
        self._new_crisp_reading_event.clear()

        self._active_event = Event()
        self._active_event.set()

        self.model = spectralvd.ReverseRFDisturbedANN.load(path)

    def run(self):
        while True:
            self._active_event.wait()
            self._new_crisp_reading_event.wait()

            predicted_rf = self.reconstruct()
            if self._active_event.is_set():
                self.new_reversal.emit(predicted_rf)
            else:
                self.active_state_changed.emit(self._active_event.is_set())
            
            self._new_crisp_reading_event.clear()

    def submit_reversal(self, rf, freqs, ff):
        self.rf, self.freqs, self.ff = rf, freqs, ff
        self._new_crisp_reading_event.set()
    
    def set_active(self, active_state):
        if active_state:
            self._active_event.set()
        else:
            self._active_event.clear()
    
    def reconstruct(self):
        prediction = self.model.predict([self.rf]*2, [(self.freqs,self.ff)]*2)
        return prediction[0]


class PeakThread(ReadThread):

    new_peaks = qtc.pyqtSignal(np.ndarray)

    def __init__(self, path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._active_event = Event()
        self._active_event.set()

        self.model = spectralvd.PeakANNTHz.load(path)

    def run(self):
        while True:
            self.wait_for_next_shot()
            self._active_event.wait()

            rf, freqs, ffs, ff_noise, detlim, charge = self.read()
            cleaned = self.clean(freqs, ffs, ff_noise, detlim, charge)

            peaks = self.model.predict(cleaned).squeeze()

            self.new_peaks.emit(peaks)
    
    def set_active(self, active_state):
        if active_state:
            self._active_event.set()
        else:
            self._active_event.clear()
    
    def read(self):
        rf_future = self.executor.submit(self.get_rf)
        charge_future = self.executor.submit(self.get_charge)
        ff_future = self.executor.submit(self.get_formfactor)
        detlim_future = self.executor.submit(self.get_detlim)
        
        rf = rf_future.result()
        charge = charge_future.result()
        freqs, ff_sq = ff_future.result()
        detlim = detlim_future.result()

        ff_sq = ff_sq.transpose()

        self.rfs.append(rf)
        self.ffs.append(ff_sq)
        self.charges.append(charge)

        n_bunches = ff_sq.shape[0]

        rf_mean = np.array(self.rfs).mean(axis=0)
        rf_mean_all = np.repeat(rf_mean[np.newaxis,:], n_bunches, axis=0)

        freqs_all = np.repeat(freqs[np.newaxis,:], n_bunches, axis=0)

        ff_all = np.zeros((n_bunches,240))
        ff_noise_all = np.zeros((n_bunches,240))
        for i in range(n_bunches):
            ffs = [x[i] for x in self.ffs]

            ff_sq_mean = np.array(ffs).mean(axis=0)
            ff = np.sqrt(np.abs(ff_sq_mean)) * np.sign(ff_sq_mean)
            ff_all[i] = ff

            ff_sq_noise = np.std(np.array(ffs), axis=0)
            ff_noise = np.abs(0.5 / ff * ff_sq_noise)
            ff_noise_all[i] = ff_noise

        detlim_mean = np.sqrt(detlim * np.sqrt(10 / len(self.ffs)))
        detlim_all = np.repeat(detlim_mean[np.newaxis,:], n_bunches, axis=0)
        
        charge_all = np.repeat(np.mean(self.charges), n_bunches)

        return rf_mean_all, freqs_all, ff_all, ff_noise_all, detlim_all, charge_all

    def get_formfactor(self):
        freqs = pydoocs.read(self.crisp_channel + "FORMFACTOR.XY")["data"][:,0] * 1e12
        ff_sq = pydoocs.read(self.crisp_channel + "FORMFACTOR.ARRAY")["data"][:,::2]
        return freqs, ff_sq
    
    def clean(self, freqs, ffs, ff_noise, detlim, charge):
        return [(freqs_clean, ff_clean) for freqs_clean, ff_clean, _ in self.executor.map(recon.cleanup_formfactor, freqs, ffs, ff_noise, detlim, [[]]*len(ffs))]


class FormfactorPlot(pg.PlotWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        pen_crisp = pg.mkPen(qtg.QColor(*COLORS[0]), width=2)
        pen_clean = pg.mkPen(qtg.QColor(*COLORS[1]), width=2)

        self.addLegend()
        self.plot_crisp = self.plot(range(999), np.ones(999), pen=pen_crisp, name="CRISP")
        self.plot_clean = self.plot(range(999), np.ones(999), pen=pen_clean, name="Cleaned")
        self.setLogMode(x=True, y=True)
        self.setLabel("bottom", text="Frequency", units="Hz")
        self.setLabel("left", text="|Frequency|")
        self.showGrid(x=True, y=True)

        self.freqs_scaled = range(999)
    
    def update(self, freqs, ff, ff_noise, detlim, charge):
        self.freqs_scaled = freqs.copy() # np.log10(frequency)
        self.ff_scaled = ff.copy()
        self.ff_scaled[self.ff_scaled <= 0] = 1e-3

        self.plot_crisp.setData(self.freqs_scaled, self.ff_scaled)
    
    def update_clean(self, freqs, ff):
        ff_scaled = np.interp(self.freqs_scaled, freqs, ff)

        self.plot_clean.setData(self.freqs_scaled, ff_scaled)


class CurrentPlot(pg.PlotWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        limit = 0.0001
        s = np.linspace(-limit, limit, 100)
        
        nils_pen = pg.mkPen(qtg.QColor(*COLORS[0]), width=2)
        lockmann_pen = pg.mkPen(qtg.QColor(*COLORS[1]), width=2)
        annrf_pen = pg.mkPen(qtg.QColor(*COLORS[2]), width=2)
        annthz_pen = pg.mkPen(qtg.QColor(*COLORS[3]), width=2)
        annrfthz_pen = pg.mkPen(qtg.QColor(*COLORS[4]), width=2)
        knnthz_pen = pg.mkPen(qtg.QColor(*COLORS[5]), width=2)
        
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


class PeakPlot(pg.PlotWidget):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        pen = pg.mkPen(qtg.QColor(0, 128, 255), width=3)
        
        self.bar = pg.BarGraphItem(x=np.arange(1024), height=np.zeros(1024), width=0.6, color="green")
        self.addItem(self.bar)
    
    def update(self, peaks):
        peaks_scaled = peaks.copy()               # * 1e6
        self.bar.setOpts(height=peaks_scaled)


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
        self.rf_checkbox = qtw.QCheckBox("RF Prediction")
        self.rf_checkbox.setChecked(True)
        self.peak_checkbox = qtw.QCheckBox("Peak Prediction")
        self.peak_checkbox.setChecked(True)

        self.l1 = qtw.QLabel("N bunch: x2 ")
        self.sb_nbunch = qtw.QSpinBox()
        self.sb_nbunch.setMaximum(1024)

        self.refresh_button = qtw.QPushButton("Refresh Low Frequencies")
        self.grating_label = qtw.QLabel("-")

        self.formfactor_plot = FormfactorPlot()
        self.current_plot = CurrentPlot()
        self.peak_plot = PeakPlot()

        self.a1v_label = qtw.QLabel("A1 Voltage")
        self.a1v_label.setStyleSheet("font-weight: bold")
        self.a1v_true = qtw.QLabel("-")
        self.a1v_predict = qtw.QLabel("-")
        self.a1phi_label = qtw.QLabel("A1 Phase")
        self.a1phi_label.setStyleSheet("font-weight: bold")
        self.a1phi_true = qtw.QLabel("-")
        self.a1phi_predict = qtw.QLabel("-")
        self.ah1v_label = qtw.QLabel("A1H Voltage")
        self.ah1v_label.setStyleSheet("font-weight: bold")
        self.ah1v_true = qtw.QLabel("-")
        self.ah1v_predict = qtw.QLabel("-")
        self.ah1phi_label = qtw.QLabel("AH1 Phase")
        self.ah1phi_label.setStyleSheet("font-weight: bold")
        self.ah1phi_true = qtw.QLabel("-")
        self.ah1phi_predict = qtw.QLabel("-")
        self.l1v_label = qtw.QLabel("L1 Voltage")
        self.l1v_label.setStyleSheet("font-weight: bold")
        self.l1v_true = qtw.QLabel("-")
        self.l1v_predict = qtw.QLabel("-")
        self.l1phi_label = qtw.QLabel("L1 Phase")
        self.l1phi_label.setStyleSheet("font-weight: bold")
        self.l1phi_true = qtw.QLabel("-")
        self.l1phi_predict = qtw.QLabel("-")
        self.l2v_label = qtw.QLabel("L2 Voltage")
        self.l2v_label.setStyleSheet("font-weight: bold")
        self.l2v_true = qtw.QLabel("-")
        self.l2v_predict = qtw.QLabel("-")
        self.l2phi_label = qtw.QLabel("L2 Phase")
        self.l2phi_label.setStyleSheet("font-weight: bold")
        self.l2phi_true = qtw.QLabel("-")
        self.l2phi_predict = qtw.QLabel("-")

        self.read_thread = ReadThread()
        self.nils_thread = NilsThread()
        self.lockmann_thread = LockmANNThread("models/annthz")
        self.annrf_thread = ANNRFThread("models/annrf")
        self.annthz_thread = ANNTHzThread("models/annthz")
        self.annrfthz_thread = ANNRFTHzThread("models/annrfthz")
        self.knnthz_thread = KNNTHzThread("models/knnthz")
        self.reverse_thread = ReverseThread("models/reverserfdisturbedann")
        self.peak_thread = PeakThread("models/peakannthz")

        self.read_thread.new_raw_reading.connect(self.formfactor_plot.update)
        self.read_thread.new_clean_reading.connect(self.formfactor_plot.update_clean)
        self.read_thread.new_rf_reading.connect(self.update_rf_true)
        self.read_thread.new_raw_reading.connect(self.nils_thread.submit_reconstruction)
        self.read_thread.new_raw_reading.connect(self.lockmann_thread.submit_reconstruction)
        self.read_thread.new_rf_reading.connect(self.annrf_thread.submit_reconstruction)
        self.read_thread.new_clean_reading.connect(self.annthz_thread.submit_reconstruction)
        self.read_thread.new_combined_reading.connect(self.annrfthz_thread.submit_reconstruction)
        self.read_thread.new_clean_reading.connect(self.knnthz_thread.submit_reconstruction)
        self.read_thread.new_combined_reading.connect(self.reverse_thread.submit_reversal)
        self.read_thread.new_stage_reading.connect(self.update_stage_label)
        
        self.nils_thread.new_reconstruction.connect(self.current_plot.update_nils)
        self.lockmann_thread.new_reconstruction.connect(self.current_plot.update_lockmann)
        self.annrf_thread.new_reconstruction.connect(self.current_plot.update_annrf)
        self.annthz_thread.new_reconstruction.connect(self.current_plot.update_annthz)
        self.annrfthz_thread.new_reconstruction.connect(self.current_plot.update_annrfthz)
        self.knnthz_thread.new_reconstruction.connect(self.current_plot.update_knnthz)
        self.reverse_thread.new_reversal.connect(self.update_rf_prediction)
        self.peak_thread.new_peaks.connect(self.peak_plot.update)

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
        self.rf_checkbox.stateChanged.connect(self.reverse_thread.set_active)
        self.peak_checkbox.stateChanged.connect(self.peak_thread.set_active)

        self.sb_nbunch.valueChanged.connect(self.read_thread.set_nbunch)

        self.refresh_button.clicked.connect(self.start_crisp_refresh)

        self.reverse_thread.active_state_changed.connect(self.show_reversed_rf)

        grid = qtw.QGridLayout()
        grid.addWidget(self.formfactor_plot, 0, 0, 32, 6)
        grid.addWidget(self.current_plot, 0, 6, 32, 6)
        grid.addWidget(self.nils_checkbox, 32, 6, 1, 1)
        grid.addWidget(self.lockmann_checkbox, 32, 7, 1, 1)
        grid.addWidget(self.annrf_checkbox, 32, 8, 1, 1)
        grid.addWidget(self.annthz_checkbox, 32, 9, 1, 1)
        grid.addWidget(self.annrfthz_checkbox, 32, 10, 1, 1)
        grid.addWidget(self.knnthz_checkbox, 32, 11, 1, 1)
        grid.addWidget(self.l1, 32, 0, 1, 1)
        grid.addWidget(self.sb_nbunch, 32, 1, 1, 1)
        grid.addWidget(self.refresh_button, 32, 3, 1, 1)
        grid.addWidget(self.grating_label, 32, 4, 1, 1)
        grid.addWidget(self.a1v_label, 0, 12, 1, 2)
        grid.addWidget(self.a1v_true, 1, 12, 1, 2)
        grid.addWidget(self.a1v_predict, 2, 12, 1, 2)
        grid.addWidget(self.a1phi_label, 4, 12, 1, 2)
        grid.addWidget(self.a1phi_true, 5, 12, 1, 2)
        grid.addWidget(self.a1phi_predict, 6, 12, 1, 2)
        grid.addWidget(self.ah1v_label, 8, 12, 1, 2)
        grid.addWidget(self.ah1v_true, 9, 12, 1, 2)
        grid.addWidget(self.ah1v_predict, 10, 12, 1, 2)
        grid.addWidget(self.ah1phi_label, 12, 12, 1, 2)
        grid.addWidget(self.ah1phi_true, 13, 12, 1, 2)
        grid.addWidget(self.ah1phi_predict, 14, 12, 1, 2)
        grid.addWidget(self.l1v_label, 16, 12, 1, 2)
        grid.addWidget(self.l1v_true, 17, 12, 1, 2)
        grid.addWidget(self.l1v_predict, 18, 12, 1, 2)
        grid.addWidget(self.l1phi_label, 20, 12, 1, 2)
        grid.addWidget(self.l1phi_true, 21, 12, 1, 2)
        grid.addWidget(self.l1phi_predict, 22, 12, 1, 2)
        grid.addWidget(self.l2v_label, 24, 12, 1, 2)
        grid.addWidget(self.l2v_true, 25, 12, 1, 2)
        grid.addWidget(self.l2v_predict, 26, 12, 1, 2)
        grid.addWidget(self.l2phi_label, 28, 12, 1, 2)
        grid.addWidget(self.l2phi_true, 29, 12, 1, 2)
        grid.addWidget(self.l2phi_predict, 30, 12, 1, 2)
        grid.addWidget(self.peak_plot, 33, 0, 3, 12)
        grid.addWidget(self.rf_checkbox, 33, 12, 1, 1)
        grid.addWidget(self.peak_checkbox, 34, 12, 1, 1)

        self.setLayout(grid)

        self.read_thread.start()
        self.nils_thread.start()
        self.lockmann_thread.start()
        self.annrf_thread.start()
        self.annthz_thread.start()
        self.annrfthz_thread.start()
        self.knnthz_thread.start()
        self.reverse_thread.start()
        self.peak_thread.start()

        # Turn off some of the plots at app startup
        self.lockmann_checkbox.setChecked(False)
        self.annrf_checkbox.setChecked(False)
        self.annrfthz_checkbox.setChecked(False)
        self.knnthz_checkbox.setChecked(False)
        self.peak_checkbox.setChecked(False)
    
    def update_rf_true(self, rf):
        self.a1v_true.setText(f"Readback = {rf[0]:.2f}")
        self.a1phi_true.setText(f"Readback = {rf[1]:.2f}")
        self.ah1v_true.setText(f"Readback = {rf[2]:.2f}")
        self.ah1phi_true.setText(f"Readback = {rf[3]:.2f}")
        self.l1v_true.setText(f"Readback = {rf[4]:.2f}")
        self.l1phi_true.setText(f"Readback = {rf[5]:.2f}")
        self.l2v_true.setText(f"Readback = {rf[6]:.2f}")
        self.l2phi_true.setText(f"Readback = {rf[7]:.2f}")
    
    def update_rf_prediction(self, rf):
        self.a1v_predict.setText(f"DANN = {rf[0]:.2f}")
        self.a1phi_predict.setText(f"DANN = {rf[1]:.2f}")
        self.ah1v_predict.setText(f"DANN = {rf[2]:.2f}")
        self.ah1phi_predict.setText(f"DANN = {rf[3]:.2f}")
        self.l1v_predict.setText(f"DANN = {rf[4]:.2f}")
        self.l1phi_predict.setText(f"DANN = {rf[5]:.2f}")
        self.l2v_predict.setText(f"DANN = {rf[6]:.2f}")
        self.l2phi_predict.setText(f"DANN = {rf[7]:.2f}")
    
    def show_reversed_rf(self, should_show):
        if not should_show:
            self.a1v_predict.setText(f"-")
            self.a1phi_predict.setText(f"-")
            self.ah1v_predict.setText(f"-")
            self.ah1phi_predict.setText(f"-")
            self.l1v_predict.setText(f"-")
            self.l1phi_predict.setText(f"-")
            self.l2v_predict.setText(f"-")
            self.l2phi_predict.setText(f"-")
    
    def start_crisp_refresh(self):
        if not hasattr(self, "crisp_refresh_executor"):
            self.crisp_refresh_executor = ThreadPoolExecutor(max_workers=1)

        self.crisp_refresh_executor.submit(self.refresh_crisp)
    
    def refresh_crisp(self):
        if self.refresh_button.isEnabled:
            self.refresh_button.isEnabled = False
            crisp.update_crisp()
            self.refresh_button.isEnabled = True
    
    def update_stage_label(self, stage):
        self.grating_label.setText(f"Grating = {stage}")

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
    window.resize(1300, 750)
    window.show()

    app.aboutToQuit.connect(window.handle_application_exit)

    sys.exit(app.exec_())
