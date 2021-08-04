import sys
from time import sleep

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.style.use("dark_background")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
import PyQt5.QtCore as qtc
import PyQt5.QtGui as qtg
import PyQt5.QtWidgets as qtw

from spectralvd import SpectralVD


class FormfactorPlot(FigureCanvasQTAgg):

    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)

        super().__init__(self.fig)

        self.plot_crisp, = self.ax.loglog(range(999), np.ones(999), label="CRISP", color="cyan")

        self.ax.set_xlim([684283010000*1e-12, 58267340000000*1e-12])
        self.ax.set_ylim([10e-3, 2])    # TODO: Better limits?
        self.ax.set_xlabel("f (THz)")
        self.ax.set_ylabel("|F|]")
        self.ax.legend(loc="upper right")

        self.fig.tight_layout()

        self.setFixedSize(650, 400)
    
    @qtc.pyqtSlot(tuple)
    def update_crisp(self, formfactor):
        frequency_scaled = formfactor[0] * 1e-12
        formfactor_scaled = formfactor[1]

        self.plot_crisp.set_xdata(frequency_scaled)
        self.plot_crisp.set_ydata(formfactor_scaled)

        self.draw()


class CurrentPlot(FigureCanvasQTAgg):

    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)

        super().__init__(self.fig)

        limit = 0.00020095917745111108 * 1e6
        s = np.linspace(-limit, limit, 100)

        self.plot_ann, = self.ax.plot(s, np.zeros(100), label="ANN", color="green")
        self.plot_nils, = self.ax.plot(s, np.zeros(100), label="Nils", color="red")
        
        self.ax.set_xlim([-limit, limit])
        self.ax.set_ylim([0, 10])
        self.ax.set_xlabel("s (μm)")
        self.ax.set_ylabel("Current (kA)")
        self.ax.legend(loc="upper right")

        self.fig.tight_layout()

        self.setFixedSize(650, 400)
    
    @qtc.pyqtSlot(tuple)
    def update_ann(self, current):
        s_scaled = current[0] * 1e6
        current_scaled = current[1] * 1e-3

        self.plot_ann.set_xdata(s_scaled)
        self.plot_ann.set_ydata(current_scaled)

        self.draw()
    
    @qtc.pyqtSlot(tuple)
    def update_nils(self, current):
        s_scaled = current[0] * 1e6
        current_scaled = current[1] * 1e-3

        self.plot_nils.set_xdata(s_scaled)
        self.plot_nils.set_ydata(current_scaled)

        self.draw()


class AcceleratorInterfaceThread(qtc.QThread):
    
    crisp_updated = qtc.pyqtSignal(tuple)
    ann_current_updated = qtc.pyqtSignal(tuple)
    nils_current_updated = qtc.pyqtSignal(tuple)

    def __init__(self):
        super().__init__()

        self.spectralvd = SpectralVD()

    def run(self):
        while True:
            self.spectralvd.read_crisp()
            self.crisp_updated.emit(self.spectralvd.crisp_reading)
            
            ann_current = self.spectralvd.ann_reconstruction()
            self.ann_current_updated.emit(ann_current)

            nils_current = self.spectralvd.nils_reconstruction()
            self.nils_current_updated.emit(nils_current)

            sleep(0.1)


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

        hbox = qtw.QHBoxLayout()
        hbox.addWidget(self.formfactor_plot)
        hbox.addWidget(self.current_plot)
        self.setLayout(hbox)

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
