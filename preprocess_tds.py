import argparse
from functools import partial
import pickle

from lmfit.models import GaussianModel
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import pandas as pd

from mint.snapshot import SnapshotDB


def reduce_to_relevant_channels(df):
    """Reduce originally recorded data to a simple and readable DataFrame of only what is needed to get current profiles from TDS."""

    is_beam_on = df["XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED"].astype("bool")
    l1_chirp_phase = df["XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L1/SUMVOLTAGE.CHIRP.SP.1"]
    charge = df["XFEL.DIAG/CHARGE.ML/TORA.25.I1/CHARGE.ALL"] * 10e-9
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


def extract_current_profile(row, shear, pixel_size):
    """Extract current profile."""

    seconds_per_pixel = pixel_size / shear

    view = row["tds_image"].sum(axis=0)
    current = row["charge"] / seconds_per_pixel * view / view.sum()
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


def preprocess_tds(tds, shear, pixel_size, plot=False):
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

    extract = partial(extract_current_profile, shear=shear, pixel_size=pixel_size)
    preprocessed = preprocessed.apply(extract, axis=1)

    if plot:
        plot_profiles(preprocessed)
    
    return preprocessed


def plot_profiles(preprocessed):
    plt.figure(figsize=(18,6))
    for i, chirp in enumerate(preprocessed.index.values):
        plt.subplot(3, 6, i+1)
        plt.title(f"Chirp = {chirp}")
        plt.plot(preprocessed[chirp][0], preprocessed[chirp][1])
        plt.xlabel("Time (s)")
        plt.ylabel("Current (A)")
    plt.tight_layout()
    plt.show()


def save_preprocessed_tds(input_path, output_path, shear, pixel_size, plot=False):
    snapshot_db = SnapshotDB(input_path)
    tds = snapshot_db.load()

    preprocessed = preprocess_tds(tds, shear, pixel_size, plot=plot)
    
    with open(output_path, "wb") as file:
        pickle.dump(preprocessed, file)

# Example call
# python3 preprocess_tds.py original/20210221-01_30_17_scan_phase1.pcl tds_feb_phase1.pkl --shear 0.0027162e-12 --pixel_size 5.435999948531389e-6 --plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess recorded TDS data to current profiles")
    parser.add_argument("input_path", type=str,
                        help="Path to the input file")
    parser.add_argument("output_path", type=str,
                        help="Path to write the preprocessed data to")
    parser.add_argument("--shear", type=float, required=True,
                        help="Longitudinal calibration in m/s")
    parser.add_argument("--pixel_size", type=float, required=True,
                        help="Width of the pixels of the camera used in m")
    parser.add_argument("--plot", default=False, action="store_true",
                        help="Show a plot of the extracted current profiles")
    args = parser.parse_args()

    save_preprocessed_tds(args.input_path,
                          args.output_path,
                          args.shear,
                          args.pixel_size,
                          plot=args.plot)
