import matplotlib.pyplot as plt

from image_proc.tds_analysis import TDSImage
from mint.snapshot import SnapshotDB


db = SnapshotDB("20210221-01_30_17_scan_phase1.pcl")
df2 = db.load()
print(db.orbit_sections)
print(df2['timestamp'])
print(df2["XFEL.DIAG/CAMERA/OTRA.473.B2D/IMAGE_EXT_ZMQ"])
print(df2["XFEL.RF/LLRF.SUMVOLTAGE_CTRL/I1/SUMVOLTAGE.CHIRP.SP.1"])

for path in df2["XFEL.DIAG/CAMERA/OTRA.473.B2D/IMAGE_EXT_ZMQ"].tolist():
    tds_img = TDSImage()
    filename = path[2:-3] + "pcl"
    tds_img.filename = filename
    print(tds_img.filename)
    tds_img.process()
    tds_img.plot()
    
x, y = db.get_orbits(section_id="B2")
x.T.plot(legend='timestamp')
plt.show()