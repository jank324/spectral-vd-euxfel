from pathlib import Path
from typing import Optional

import lightning as L
import numpy as np
import ocelot as oc
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

from utils import current2formfactor


class EuXFELCurrentDataset(Dataset):
    """
    Dataset on reconstructing current profiles from RF settings and THz spectrum
    formfactors.

    `X` is RF settings and the formfactors. `y` is the current profiles and bunch
    lengths.
    """

    def __init__(
        self,
        stage: str = "train",
        normalize: bool = False,
        rf_scaler: Optional[StandardScaler] = None,
        formfactor_scaler: Optional[StandardScaler] = None,
        current_scaler: Optional[StandardScaler] = None,
        bunch_length_scaler: Optional[StandardScaler] = None,
    ):
        self.normalize = normalize

        assert stage in ["train", "validation", "test"]

        df = pd.read_pickle(f"data/zihan/data_20220905_{stage}.pkl")

        self.current_profiles = np.stack(df.loc[:, "slice_I"])
        self.rf_settings = df.loc[
            :, ["chirp", "curv", "skew", "chirpL1", "chirpL2"]
        ].values
        self.bunch_lengths = np.expand_dims(
            df.loc[:, "slice_width"].values * self.current_profiles.shape[1], axis=1
        )

        self.formfactors = [
            current2formfactor(
                ss=np.linspace(0, bunch_length[0], num=len(current_profile)),
                currents=np.array(current_profile),
                grating="both",
                clean=False,
                n_shots=10,
            )[1]
            for current_profile, bunch_length in zip(
                self.current_profiles, self.bunch_lengths
            )
        ]

        if self.normalize:
            self.setup_normalization(
                rf_scaler, formfactor_scaler, current_scaler, bunch_length_scaler
            )

    def __len__(self):
        return len(self.formfactors)

    def __getitem__(self, index):
        rf_settings = self.rf_settings[index]
        formfactor = self.formfactors[index]
        current_profile = self.current_profiles[index]
        bunch_length = self.bunch_lengths[index]

        if self.normalize:
            rf_settings = self.rf_scaler.transform([rf_settings])[0]
            formfactor = self.formfactor_scaler.transform([formfactor])[0]
            current_profile = self.current_scaler.transform([current_profile])[0]
            bunch_length = self.bunch_length_scaler.transform([bunch_length])[0]

        rf_settings = torch.tensor(rf_settings, dtype=torch.float32)
        formfactor = torch.tensor(formfactor, dtype=torch.float32)
        current_profile = torch.tensor(current_profile, dtype=torch.float32)
        bunch_length = torch.tensor(bunch_length, dtype=torch.float32)

        return (rf_settings, formfactor), (current_profile, bunch_length)

    def setup_normalization(
        self,
        rf_scaler: Optional[StandardScaler] = None,
        formfactor_scaler: Optional[StandardScaler] = None,
        current_scaler: Optional[MinMaxScaler] = None,
        bunch_length_scaler: Optional[MinMaxScaler] = None,
    ) -> None:
        """
        Creates normalisation scalers for each of the four variables return by this
        dataset. Pass scalers that should be used. If a scaler is not passed, a new one
        is fitted to the data in the dataset.
        """

        self.rf_scaler = (
            rf_scaler
            if rf_scaler is not None
            else StandardScaler().fit(self.rf_settings)
        )

        self.formfactor_scaler = (
            formfactor_scaler
            if formfactor_scaler is not None
            else StandardScaler().fit(self.formfactors)
        )

        current_profiles_including_zero = np.concatenate(
            [np.zeros([1, 300]), self.current_profiles], axis=0
        )  # Include zero to make sure in the scaler min = 0
        self.current_scaler = (
            current_scaler
            if current_scaler is not None
            else MinMaxScaler().fit(current_profiles_including_zero)
        )

        bunch_lengths_including_zero = np.concatenate(
            [np.zeros([1, 1]), self.bunch_lengths], axis=0
        )  # Include zero to make sure in the scaler min = 0
        self.bunch_length_scaler = (
            bunch_length_scaler
            if bunch_length_scaler is not None
            else MinMaxScaler().fit(bunch_lengths_including_zero)
        )


class EuXFELCurrentDataModule(L.LightningDataModule):
    """
    Data Module for reconstructing current profiles from RF settings and THz spectrum
    formfactors.
    """

    def __init__(self, batch_size=32, normalize=False, num_workers=10):
        super().__init__()
        self.batch_size = batch_size
        self.normalize = normalize
        self.num_workers = num_workers

    def setup(self, stage):
        self.dataset_train = EuXFELCurrentDataset(
            stage="train", normalize=self.normalize
        )
        self.dataset_val = EuXFELCurrentDataset(
            stage="validation",
            normalize=self.normalize,
            rf_scaler=self.dataset_train.rf_scaler,
            formfactor_scaler=self.dataset_train.formfactor_scaler,
            current_scaler=self.dataset_train.current_scaler,
            bunch_length_scaler=self.dataset_train.bunch_length_scaler,
        )
        self.dataset_test = EuXFELCurrentDataset(
            stage="test",
            normalize=self.normalize,
            rf_scaler=self.dataset_train.rf_scaler,
            formfactor_scaler=self.dataset_train.formfactor_scaler,
            current_scaler=self.dataset_train.current_scaler,
            bunch_length_scaler=self.dataset_train.bunch_length_scaler,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers
        )


class EuXFELLPSDataset(Dataset):
    """
    Dataset on reconstructing longitudinal phase spaces from RF settings and THz
    spectrum formfactors.

    `X` is RF settings and the formfactors. `y` is the longitudinal phase space density
    image and its dimensions (bunch length and energy spread).

    NOTE: As it stands now LPS images must be transposed for correct plotting with s on
    x and energy spread on y.
    """

    particle_dir = Path("data/zihan/particles")

    def __init__(
        self,
        stage: str = "train",
        normalize: bool = False,
        rf_scaler: Optional[StandardScaler] = None,
        formfactor_scaler: Optional[StandardScaler] = None,
        lps_image_scaler: Optional[StandardScaler] = None,
        lps_range_scaler: Optional[StandardScaler] = None,
    ):
        self.normalize = normalize

        assert stage in ["train", "validation", "test"]

        df = pd.read_pickle(f"data/zihan/data_20220905_{stage}.pkl")

        current_profiles = np.stack(df.loc[:, "slice_I"])
        bunch_lengths = np.expand_dims(
            df.loc[:, "slice_width"].values * current_profiles.shape[1], axis=1
        )

        self.rf_settings = df.loc[
            :, ["chirp", "curv", "skew", "chirpL1", "chirpL2"]
        ].values
        self.formfactors = [
            current2formfactor(
                ss=np.linspace(0, bunch_length[0], num=len(current_profile)),
                currents=np.array(current_profile),
                grating="both",
                clean=False,
                n_shots=10,
            )[1]
            for current_profile, bunch_length in zip(current_profiles, bunch_lengths)
        ]
        self.particle_distribution_files = df["file"].values.tolist()
        self.lps_images, self.lps_ranges = self.load_all_lps_images_and_ranges()

        if self.normalize:
            self.setup_normalization(
                rf_scaler, formfactor_scaler, lps_image_scaler, lps_range_scaler
            )

    def __len__(self):
        return len(self.formfactors)

    def __getitem__(self, index):
        rf_settings = self.rf_settings[index]
        formfactor = self.formfactors[index]

        lps_image = self.lps_images[index]
        lps_range = self.lps_ranges[index]

        if self.normalize:
            rf_settings = self.rf_scaler.transform([rf_settings])[0]
            formfactor = self.formfactor_scaler.transform([formfactor])[0]
            lps_image = self.lps_image_scaler.transform([lps_image.flatten()])[
                0
            ].reshape(300, 300)
            lps_range = self.lps_range_scaler.transform([lps_range])[0]

        rf_settings = torch.tensor(rf_settings, dtype=torch.float32)
        formfactor = torch.tensor(formfactor, dtype=torch.float32)
        lps_image = torch.tensor(lps_image, dtype=torch.float32)
        lps_range = torch.tensor(lps_range, dtype=torch.float32)

        return (rf_settings, formfactor), (lps_image, lps_range)

    def load_all_lps_images_and_ranges(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads the dataset's particle distribution files, computes the longitudinal phase
        spaces from them, and then returns the phase space images and their ranges.
        """
        lps_images = []
        lps_ranges = []
        for index in range(len(self)):
            lps_image, lps_range = self.load_lps_image(index)
            lps_images.append(lps_image)
            lps_ranges.append(lps_range)

        return np.array(lps_images), np.array(lps_ranges)

    def load_lps_image(self, index) -> tuple[np.ndarray, tuple[float, float]]:
        """
        Loads the particle distribution file for sample `index`, compute the
        longitudinal phase spaces from it, and then returns the phase space image and
        its ranges.
        """
        filepath = self.particle_dir / self.particle_distribution_files[index]

        particle_array = oc.load_particle_array(filepath)
        lps_image, x_edges, y_edges = np.histogram2d(
            x=particle_array.tau(), y=particle_array.p(), bins=300
        )
        s_range = x_edges.max() - x_edges.min()
        energy_spread_range = y_edges.max() - y_edges.min()

        lps_range = (s_range, energy_spread_range)

        return lps_image, lps_range

    def setup_normalization(
        self,
        rf_scaler: Optional[StandardScaler] = None,
        formfactor_scaler: Optional[StandardScaler] = None,
        lps_image_scaler: Optional[MinMaxScaler] = None,
        lps_range_scaler: Optional[MinMaxScaler] = None,
    ) -> None:
        """
        Creates normalisation scalers for each of the four variables return by this
        dataset. Pass scalers that should be used. If a scaler is not passed, a new one
        is fitted to the data in the dataset.
        """

        self.rf_scaler = (
            rf_scaler
            if rf_scaler is not None
            else StandardScaler().fit(self.rf_settings)
        )

        self.formfactor_scaler = (
            formfactor_scaler
            if formfactor_scaler is not None
            else StandardScaler().fit(self.formfactors)
        )

        lps_images_including_zero = np.concatenate(
            [np.zeros([1, 300, 300]), self.lps_images], axis=0
        )  # Include zero to make sure in the scaler min = 0
        self.lps_image_scaler = (
            lps_image_scaler
            if lps_image_scaler is not None
            else MinMaxScaler().fit(lps_images_including_zero.reshape(-1, 300 * 300))
        )

        lps_ranges_including_zero = np.concatenate(
            [np.zeros([1, 2]), self.lps_ranges], axis=0
        )  # Include zero to make sure in the scaler min = 0
        self.lps_range_scaler = (
            lps_range_scaler
            if lps_range_scaler is not None
            else MinMaxScaler().fit(lps_ranges_including_zero)
        )


class EuXFELLPSDataModule(L.LightningDataModule):
    """
    Data Module for reconstructing longitudinal phase spaces from RF settings and THz
    spectrum formfactors.
    """

    def __init__(self, batch_size=32, normalize=False, num_workers=10):
        super().__init__()
        self.batch_size = batch_size
        self.normalize = normalize
        self.num_workers = num_workers

    def setup(self, stage):
        self.dataset_train = EuXFELLPSDataset(stage="train", normalize=self.normalize)
        self.dataset_val = EuXFELLPSDataset(
            stage="validation",
            normalize=self.normalize,
            rf_scaler=self.dataset_train.rf_scaler,
            formfactor_scaler=self.dataset_train.formfactor_scaler,
            lps_image_scaler=self.dataset_train.lps_image_scaler,
            lps_range_scaler=self.dataset_train.lps_range_scaler,
        )
        self.dataset_test = EuXFELLPSDataset(
            stage="test",
            normalize=self.normalize,
            rf_scaler=self.dataset_train.rf_scaler,
            formfactor_scaler=self.dataset_train.formfactor_scaler,
            lps_image_scaler=self.dataset_train.lps_image_scaler,
            lps_range_scaler=self.dataset_train.lps_range_scaler,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
