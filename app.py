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


class CurrentPlot(FigureCanvasQTAgg):

    def __init__(self):
        self.fig = Figure()
        self.ax0 = self.fig.add_subplot(111)

        super().__init__(self.fig)

        self.plot_ann, = self.ax0.plot(range(100), np.zeros(100), label="ANN", color="green")
        self.plot_nils, = self.ax0.plot(range(100), np.zeros(100), label="Nils", color="red")

        self.ax0.set_ylim([0, 10])
        self.ax0.set_xlabel("s (μm)")
        self.ax0.set_ylabel("Current (kA)")
        self.ax0.legend()

        self.fig.tight_layout()

        self.setFixedSize(650, 400)
    
    @qtc.pyqtSlot(tuple)
    def update_ann(self, current):
        s_scaled = current[0] * 1e6
        current_scaled = current[1] * 1e-3

        self.plot_ann.set_xdata(s_scaled)
        self.plot_ann.set_ydata(current_scaled)

        self.ax0.set_xlim([s_scaled.min(), s_scaled.max()])
        self.ax0.set_ylim([0, 10])

        self.draw()
    
    @qtc.pyqtSlot(tuple)
    def update_nils(self, current):
        s_scaled = current[0] * 1e6
        current_scaled = current[1] * 1e-3

        self.plot_nils.set_xdata(s_scaled)
        self.plot_nils.set_ydata(current_scaled)

        # self.ax0.set_xlim([current[0].min(), current[0].max()])
        # self.ax0.set_ylim([0, max(1, current[1].max())])

        self.draw()


class AcceleratorInterfaceThread(qtc.QThread):
    
    ann_current_updated = qtc.pyqtSignal(tuple)
    nils_current_updated = qtc.pyqtSignal(tuple)

    def __init__(self):
        super().__init__()

        self.spectralvd = SpectralVD()

    def run(self):
        while True:
            crisp = self.spectralvd.read_crisp()
            
            ann_current = self.spectralvd.ann_reconstruction(crisp)
            self.ann_current_updated.emit(ann_current)

            nils_current = self.spectralvd.nils_reconstruction(crisp)
            self.nils_current_updated.emit(nils_current)

            sleep(0.1)
    
    def change_grating(self, grating):
        print(f"Changing grating to {grating}")


class App(qtw.QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Spectral Virtual Diagnostics at European XFEL")

        self.current_plot = CurrentPlot()

        self.interface_thread = AcceleratorInterfaceThread()
        self.interface_thread.ann_current_updated.connect(self.current_plot.update_ann)
        self.interface_thread.nils_current_updated.connect(self.current_plot.update_nils)

        self.grating_dropdown = qtw.QComboBox()
        self.grating_dropdown.addItems(["low", "high", "both"])
        self.grating_dropdown.currentTextChanged.connect(self.interface_thread.change_grating)

        hbox = qtw.QHBoxLayout()
        hbox.addWidget(self.current_plot)
        vbox = qtw.QVBoxLayout()
        vbox.addWidget(qtw.QLabel("Set Grating"))
        vbox.addWidget(self.grating_dropdown)
        vbox.addStretch()
        hbox.addLayout(vbox)
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
