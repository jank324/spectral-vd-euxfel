import sys
from threading import Event

import numpy as np
import PyQt5.QtCore as qtc
import PyQt5.QtGui as qtg
import PyQt5.QtWidgets as qtw
import pyqtgraph as pg

from spectralvd import SpectralVD


class FormfactorPlot(pg.PlotWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        pen = pg.mkPen("c", width=2)
        self.plot_crisp = self.plot(range(999), np.ones(999), pen=pen, name="CRISP")
        # self.setXRange(int(684283010000), int(58267340000000))
        # self.setYRange(10e-3, 2)
        # self.setLogMode(x=True, y=True)
        self.setLabel("bottom", text="Frequency", units="Hz")
        self.setLabel("left", text="|Frequency|")
        self.addLegend()
        self.showGrid(x=True, y=True)
    
    @qtc.pyqtSlot(tuple)
    def update_crisp(self, formfactor):
        frequency_scaled = np.log10(formfactor[0])
        formfactor_scaled = formfactor[1].copy()
        formfactor_scaled[formfactor_scaled < 0] = 0
        formfactor_scaled = np.log10(formfactor_scaled + 1)

        self.plot_crisp.setData(frequency_scaled, formfactor_scaled)


class CurrentPlot(pg.PlotWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        limit = 0.00020095917745111108 # * 1e6
        s = np.linspace(-limit, limit, 100)

        ann_pen = pg.mkPen("g", width=2)
        nils_pen = pg.mkPen("r", width=2)
        self.ann_plot = self.plot(s, np.zeros(100), pen=ann_pen, name="ANN")
        self.nils_plot = self.plot(s, np.zeros(100), pen=nils_pen, name="Nils")

        self.setXRange(-limit, limit)
        self.setYRange(0, 10e3)
        self.setLabel("bottom", text="s", units="m")
        self.setLabel("left", text="Current", units="A")
        self.addLegend()
        self.showGrid(x=True, y=True)
    
    @qtc.pyqtSlot(tuple)
    def update_ann(self, current):
        s_scaled = current[0] # * 1e6
        current_scaled = current[1] # * 1e-3

        self.ann_plot.setData(s_scaled, current_scaled)
    
    @qtc.pyqtSlot(tuple)
    def update_nils(self, current):
        s_scaled = current[0] # * 1e6
        current_scaled = current[1] # * 1e-3

        self.nils_plot.setData(s_scaled, current_scaled)


class AcceleratorInterfaceThread(qtc.QThread):
    
    crisp_updated = qtc.pyqtSignal(tuple)
    ann_current_updated = qtc.pyqtSignal(tuple)
    nils_current_updated = qtc.pyqtSignal(tuple)

    def __init__(self):
        super().__init__()

        self.spectralvd = SpectralVD()
        
        self.running_event = Event()
        self.running_event.clear()

    def run(self):
        while True:
            self.running_event.wait()

            self.spectralvd.read_crisp()
            ann_current = self.spectralvd.ann_reconstruction()
            nils_current = self.spectralvd.nils_reconstruction()

            self.crisp_updated.emit(self.spectralvd.crisp_reading)
            self.ann_current_updated.emit(ann_current)
            self.nils_current_updated.emit(nils_current)
        
    def toggle_running(self):
        if self.running_event.is_set():
            self.running_event.clear()
        else:
            self.running_event.set()


class App(qtw.QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Spectral Virtual Diagnostics at European XFEL")

        self.formfactor_plot = FormfactorPlot()

        self.current_plot = CurrentPlot()

        self.interface_thread = AcceleratorInterfaceThread()
        self.interface_thread.crisp_updated.connect(self.formfactor_plot.update_crisp)
        self.interface_thread.ann_current_updated.connect(self.current_plot.update_ann)
        self.interface_thread.nils_current_updated.connect(self.current_plot.update_nils)

        self.start_stop_button = qtw.QPushButton("Start/Stop")
        self.start_stop_button.clicked.connect(self.interface_thread.toggle_running)

        grid = qtw.QGridLayout()
        grid.addWidget(self.formfactor_plot, 0, 0, 1, 3)
        grid.addWidget(self.current_plot, 0, 3, 1, 3)
        grid.addWidget(self.start_stop_button, 1, 2, 1, 2)
        self.setLayout(grid)

        self.interface_thread.start()
    
    def handle_application_exit(self):
        print("Handling application exit")
    

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
