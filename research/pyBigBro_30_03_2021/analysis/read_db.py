import sys
sys.path.append("/home/xfeloper/user/tomins/ocelot_test/")
import os
import pandas as pd
import matplotlib.pyplot as plt
from pyBigBro.mint.snapshot import *
from pyBigBro.image_proc.tds_analysis import *

db = SnapshotDB("/home/xfeloper/user/tomins/ocelot_test/pyBigBro/20200124-16_12_46_test.pcl")
df2 = db.load()
print(db.orbit_sections)
print(df2['BPMG.24.I1.X'])
print(df2['timestamp'])
print(df2["XFEL.DIAG/CAMERA/OTRA.473.B2D/IMAGE_EXT_ZMQ"])
print(df2["XFEL.RF/LLRF.SUMVOLTAGE_CTRL/I1/SUMVOLTAGE.CHIRP.SP.1"])

for path in df2["XFEL.DIAG/CAMERA/OTRA.473.B2D/IMAGE_EXT_ZMQ"].tolist():
    tds_img = TDSImage()
    filename = "/home/xfeloper/user/tomins/ocelot_test/pyBigBro/" + path[2:-3] + "pcl"
    tds_img.filename = filename
    print(tds_img.filename)
    tds_img.process()
    tds_img.plot()
    
x, y = db.get_orbits(section_id="B2")
x.T.plot(legend='timestamp')
plt.show()