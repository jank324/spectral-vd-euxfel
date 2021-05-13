import pickle

from lmfit.models import GaussianModel
import numpy as np
from scipy import ndimage

from mint.snapshot import SnapshotDB


def load_tds_image(path):
    with open("original/" + path[2:-4] + ".pcl", "rb") as file:
        image = pickle.load(file)
    return image


def denoise(image):
    """Denoise `image`."""
    denoised = ndimage.uniform_filter(image, size=12)
    denoised[denoised < 0.05 * denoised.max()] = 0
    return denoised


def extract_current_profile(tds_image, charge, shear, pixel_size):
    """Extract current profile."""

    seconds_per_pixel = pixel_size / shear

    view = tds_image.sum(axis=0)
    current = charge / seconds_per_pixel * view / view.sum()
    times = np.arange(len(current)) * seconds_per_pixel

    xs = np.arange(len(current))
    model = GaussianModel()
    guess = model.guess(current, x=xs)
    fit = model.fit(current, guess, x=xs)

    a = int(fit.params["center"].value - 5 * fit.params["sigma"].value)
    b = int(fit.params["center"].value + 5 * fit.params["sigma"].value)
    extracted = current[a:b]
    extracted_times = times[a:b] - fit.params["center"].value * seconds_per_pixel

    return np.vstack((extracted_times, extracted))


def load_current_profiles(path, shear, pixel_size):
    snapshot_db = SnapshotDB(path)
    raw = snapshot_db.load()

    data = raw[["XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED",
                "XFEL.DIAG/CAMERA/OTRA.473.B2D/IMAGE_EXT_ZMQ",
                "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L1/SUMVOLTAGE.CHIRP.SP.1",
                "XFEL.DIAG/CHARGE.ML/TORA.25.I1/CHARGE.ALL"]]

    data = data.rename(columns={"XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED": "beam_allowed",
                                "XFEL.DIAG/CAMERA/OTRA.473.B2D/IMAGE_EXT_ZMQ": "tds_image_path",
                                "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L1/SUMVOLTAGE.CHIRP.SP.1": "l1_chirp_phase",
                                "XFEL.DIAG/CHARGE.ML/TORA.25.I1/CHARGE.ALL": "charge"})

    data["tds_image"] = data["tds_image_path"].apply(load_tds_image)
    data["beam_allowed"] = data["beam_allowed"].astype(bool)
    data["charge"] = data["charge"] * 10e-9

    # Get background
    background = data.loc[1:6,"tds_image"].mean()

    # Average charges and images
    charges = data[6:].groupby("l1_chirp_phase")["charge"].mean()

    average_images = lambda images: np.stack(images).mean(axis=0)
    tds_images = data[6:].groupby("l1_chirp_phase")["tds_image"].apply(average_images)

    tds_images = tds_images.apply(lambda image: image.clip(0, 4095))

    # Remove background
    remove_background = lambda image: (image - background).clip(0, 4095)
    tds_images = tds_images.apply(remove_background)

    tds_images = tds_images.apply(denoise)

    current_profiles = [extract_current_profile(tds_images[chirp], charges[chirp], shear, pixel_size) for chirp in tds_images.index]

    return current_profiles
