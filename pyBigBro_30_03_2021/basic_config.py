"""
Sergey Tomin

Example of the configuration file for collecting machine data
"""
import time
from mint.machine import Machine
from mint.snapshot import Snapshot, SnapshotDB


snapshot = Snapshot()
snapshot.sase_sections = []
snapshot.magnet_prefix = None

# in case channel is out of range the Snapshot object gives a flag and measurement can be stopped
# can be a few alarm channels
snapshot.add_alarm_channels("XFEL.DIAG/TOROID/TORA.60.I1/CHARGE.ALL", min=0.005, max=0.5)

# add orbit section: all BPMs (x and y beam positions) with prefix "I1" will be saved
snapshot.add_orbit_section("I1", tol=0.1, track=False)
snapshot.add_orbit_section("I1D", tol=0.1, track=False)

# add camera channels.  Images (raw data) will be saved to specific folder
snapshot.add_image("XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ", folder="./tds_images")

# add magnet sections.
snapshot.add_magnet_section("T1", tol=0.01)

# add channel
snapshot.add_channel("XFEL.MAGNETS/MAGNET.ML/SOLB.23.I1/CURRENT.SP")
snapshot.add_channel("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/LH/ENERGY.ALL", tol=0.2)