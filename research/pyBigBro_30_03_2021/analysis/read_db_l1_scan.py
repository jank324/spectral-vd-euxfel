import sys

sys.path.append("/home/xfeloper/user/tomins/ocelot_new/")
import os
import pandas as pd
import matplotlib.pyplot as plt
from pyBigBro.mint.snapshot import *
#from pyBigBro.image_proc.tds_analysis import *
from pathlib import Path
import numpy as np
import matplotlib
from scipy import ndimage

from skimage import data
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform

font = {'size': 16}

matplotlib.rc('font', **font)


print(os.path.realpath(__file__))
print(os.path.dirname(os.path.realpath(__file__)))
db = SnapshotDB("/home/xfeloper/user/tomins/ocelot_new/pyBigBro/20200517-02_29_31_L1_ampl_no_comp.pcl")
# db = SnapshotDB("/Users/tomins/Nextcloud/Machine_studies/pyBigBro/20200126-14_11_19_l1_phase2.pcl")

# db = SnapshotDB("/Users/tomins/Nextcloud/Machine_studies/pyBigBro/20200126-12_33_46_l2_crisp_stage_2_shots_20.pcl")

df2 = db.load()
db.plot(x="XFEL.RF/LLRF.CONTROLLER/CTRL.A2.L1/SP.PHASE",
                         y=["XFEL.RF/LLRF.CONTROLLER/CTRL.A2.L1/SP.AMPL"], beam_on=True, start_inx=2)


# df2.plot(x="XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L2/SUMVOLTAGE.CHIRP.SP.1", y="XFEL.SDIAG/BCM/BCM.416.B2/PYRO.SA1")
# plt.show()
# db.plot(x="XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L2/SUMVOLTAGE.CHIRP.SP.1", y="XFEL.SDIAG/BCM/BCM.416.B2/MCT.SA1")

# db.plot(x="XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L2/SUMVOLTAGE.CHIRP.SP.1", y="XFEL.SDIAG/BCM/BCM.416.B2/PYRO.SA1",
#        beam_on=True)
#print(df2)

#print(db.orbit_sections.keys())
#db.plot_orbits(section_ids=["I1", "L1", "B1"],  legend_item="XFEL.RF/LLRF.CONTROLLER/CTRL.A2.L1/SP.PHASE",
#              subtract_first=True, halve=True)

db.plot_orbits(section_ids=["I1", "L1", "B1"],  legend_item="XFEL.RF/LLRF.CONTROLLER/CTRL.A2.L1/SP.AMPL",
              subtract_first=True, halve=False)

plt.show()

