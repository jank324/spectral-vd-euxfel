import argparse
from functools import partial
import pickle

from lmfit.models import GaussianModel
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import pandas as pd

from mint.snapshot import SnapshotDB


def load(path):
    """Load data in Sergey's `SnapshotDB` format."""
    snapshot_db = SnapshotDB(path)
    df = snapshot_db.load()
    return df


def reduce_to_relevant_channels(df):
    """Reduce originally recorded data to a simple and readable DataFrame of only what is needed to get current profiles from TDS."""

    is_beam_on = df["XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED"].astype("bool")
    l1_chirp_phase = df["XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L1/SUMVOLTAGE.CHIRP.SP.1"]
    charge = df["XFEL.DIAG/CHARGE.ML/TORA.25.I1/CHARGE.ALL"]
    tds_image = df["XFEL.DIAG/CAMERA/OTRA.473.B2D/IMAGE_EXT_ZMQ"].apply(lambda path: pickle.load(open("original/"+path[2:-4]+".pcl", "rb")))

    reduced = pd.DataFrame({"is_beam_on": is_beam_on,
                            "l1_chirp_phase": l1_chirp_phase,
                            "charge": charge,
                            "tds_image": tds_image})
    return reduced


def denoise(image):
    """Denoise `image`."""
    denoised = ndimage.uniform_filter(image, size=12)
    denoised[denoised < 0.05 * denoised.max()] = 0
    return denoised


def extract_current_profile(row, seconds_per_pixel):
    """Extract current profile."""

    view = row["tds_image"].sum(axis=0)
    current = row["charge"] * 10e-9 / seconds_per_pixel * view / view.sum()

    xs = np.arange(len(current))
    model = GaussianModel()
    guess = model.guess(current, x=xs)
    fit = model.fit(current, guess, x=xs)

    low = int(fit.params["center"].value - 4 * fit.params["sigma"].value)
    high = int(fit.params["center"].value + 4 * fit.params["sigma"].value)
    extracted = current[low:high]

    return extracted


def preprocess_tds(tds, seconds_per_pixel, plot=False):
    tds_reduced = reduce_to_relevant_channels(tds)

    background = tds_reduced.loc[1:6,"tds_image"].mean()

    preprocessed = tds_reduced.loc[6:,["l1_chirp_phase","charge","tds_image"]].copy()

    # Average all five images per chirp setting
    preprocessed = preprocessed.groupby("l1_chirp_phase").apply(np.mean).drop("l1_chirp_phase", axis=1)
    preprocessed["tds_image"] = preprocessed["tds_image"].apply(lambda image: image.clip(0, 4095))

    # Remove background
    remove_background = lambda image: (image - background).clip(0, 4095)
    preprocessed["tds_image"] = preprocessed["tds_image"].apply(remove_background)

    preprocessed["tds_image"] = preprocessed["tds_image"].apply(denoise)

    extract = partial(extract_current_profile, seconds_per_pixel=seconds_per_pixel)
    preprocessed = preprocessed.apply(extract, axis=1)

    if plot:
        plot_profiles(preprocessed)
    
    return preprocessed


def plot_profiles(preprocessed):
    plt.figure(figsize=(18,6))
    for i, chirp in enumerate(preprocessed.index.values):
        plt.subplot(3, 6, i+1)
        plt.title(f"Chirp = {chirp}")
        plt.plot(preprocessed[chirp])
    plt.tight_layout()
    plt.show()


def save_preprocessed_tds(input_path, output_path, seconds_per_pixel, plot=False):
    tds = load(input_path)
    preprocessed = preprocess_tds(tds, seconds_per_pixel, plot=plot)
    with open(output_path, "wb") as file:
        pickle.dump(preprocessed, file)

# Example call
# python3 preprocess_tds.py original/20210221-01_30_17_scan_phase1.pcl tds_feb_phase1.pkl --seconds_per_pixel 4.38912e-13 --plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess recorded TDS data to current profiles")
    parser.add_argument("input_path", type=str,
                        help="Path to the input file")
    parser.add_argument("output_path", type=str,
                        help="Path to write the preprocessed data to")
    parser.add_argument("--seconds_per_pixel", type=float, required=True,
                        help="Calibration for seconds per pixel in the TDS images")
    parser.add_argument("--plot", default=False, action="store_true",
                        help="Show a plot of the extracted current profiles")
    args = parser.parse_args()

    save_preprocessed_tds(args.input_path, args.output_path, args.seconds_per_pixel, plot=args.plot)
