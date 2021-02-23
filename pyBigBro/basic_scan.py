"""
Simple example for scanning of a DOOCS channel and collecting data which was configured

@author: S.Tomin
"""
import pydoocs
import numpy as np
import matplotlib.pyplot as plt
import time
from mint.machine import Machine
from mint.snapshot import Snapshot, SnapshotDB
import basic_config
import time
from tds_images.tds_analysis import *

# DOOCS channel to scan. Fake channel
doocs_ch = "XFEL/DOOCS/CHANNEL/A.SP"

# read initial value of the doocs channel
x_0 = pydoocs.read(doocs_ch)["data"]

# scan range
x_range = np.arange(-30., 30., 2)

# define Snapshot config
snapshot = basic_config.snapshot

# init Machine
machine = Machine(snapshot)
# init DataBase
db = SnapshotDB(filename="simple_scan" + time.strftime("%Y%m%d-%H_%M_%S") + ".pcl")

# get machine snapshot before scan
df_ref = machine.get_machine_snapshot()
db.add(df_ref)

# scanning loop
for x_i in x_range:
    # set new value
    print("set: {} <-- {}".format(doocs_ch, x_i))
    pydoocs.write(doocs_ch, x_i)
    # sleep 1 sec. Here maybe we need to wait a bit longer in some cases. Especially if the images are saved
    time.sleep(1)

    while True:
        # check if machine is online. Looking to snapshot alarm channels
        if machine.is_machine_online():
            break
        else:
            # if is not online sleep for 3 sec
            time.sleep(3)
            print("sleep 3 sec ..")
    # get machine snapshot
    df = machine.get_machine_snapshot()
    # add data to DataBase db
    db.add(df)
    time.sleep(1)

db.save()

pydoocs.write(doocs_ch, x_0)

# plotting

# plot one doocs channel against another
db.plot(x=doocs_ch, y=["XFEL.MAGNETS/MAGNET.ML/SOLB.23.I1/CURRENT.SP",
                        "XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/LH/ENERGY.ALL"])

# plot orbit with labeling by parameter if scanning channel
db.plot_orbit(section_id="I1",  legend_item=doocs_ch)

# images

parent_dir = "/../../path to directory where are images"

bg_img = db.get_background(parent_dir, cam_channel="XFEL.DIAG/CAMERA/TDS_CAMERA")

image_dict = db.get_data(x=doocs_ch, y="XFEL.DIAG/CAMERA/TDS_CAMERA", beam_on=True, calc_moments=False)

tds_img = TDSImage()
tds_img.dx_px = 5.436e-6 / 0.0038       # ps / px, pixel size in horizontal direction
tds_img.dy_px = 1.0 / 49.1              # MeV / px, pixel size in vertical direction
tds_img.noise_thresh = 0.1
tds_img.unif_filter_size = 150
tds_img.noise_proj_thresh = 0.1
tds_img.shear = -0.055 * 0
tds_img.fliplr = False
tds_img.bg_image = bg_img


# extract beam current, length, ... from TDS images and show processed images if needed
data_extract = extract_parameters(image_dict, parent_dir, tds_img, show_img=True)

