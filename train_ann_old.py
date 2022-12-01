from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import Dataset

from .utils import current2formfactor


class LengthReconstructionDataset(Dataset):
    """
    Dataset for reconstructing the bunch length from RF settings and the THz spectrum.
    """

    def __init__(
        self,
        path: Union[Path, str],
        normalize_rf: bool = True,
        normalize_formfactors: bool = True,
        normalize_lengths: bool = True,
    ) -> None:
        self.normalize_rf = normalize_rf
        self.normalize_formfactors = normalize_formfactors
        self.normalize_lengths = normalize_lengths

        self.rf_settings, self.formfactors, self.bunch_lengths_m = self.load_data(path)

        if self.normalize_rf:
            self.rf_scaler = MinMaxScaler().fit(self.rf_settings)
        if self.normalize_formfactors:
            self.formfactor_scaler = MinMaxScaler().fit(self.formfactors)
        if self.normalize_lengths:
            self.length_scaler = MinMaxScaler().fit(self.bunch_lengths_m)

    def load_data(
        self, path: Union[Path, str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data saved at `path` from disk and return the parts of it relevant to this
        dataset.
        """
        path = Path(path) if isinstance(path, str) else path

        df = pd.read_pickle(path)
        rf_settings = df[["chirp", "chirpL1", "chirpL2", "curv", "skew"]].values

        ss = np.stack(
            [np.linspace(0, 300 * df.loc[i, "slice_width"], num=300) for i in df.index]
        )
        bunch_lengths_m = ss.max(axis=1) - ss.min()

        currents = np.stack(df["slice_I"].values)

        formfactors = np.array(
            [
                current2formfactor(
                    ss, currents, grating="both", n_shots=1, clean=False
                )[1]
                for ss, currents in zip(ss, currents)
            ]
        )

        return rf_settings, formfactors, bunch_lengths_m

    def __len__(self) -> int:
        return len(self.bunch_lengths_m)

    def __getitem__(self, index: int) -> tuple[tuple[np.ndarray, np.ndarray], float]:
        rf_setting = (
            self.rf_scaler.transform(self.rf_settings[index])
            if self.normalize_rf
            else self.rf_settings[index]
        )
        formfactor = (
            self.formfactor_scaler.transform(self.formfactors[index])
            if self.normalize_formfactors
            else self.formfactors[index]
        )
        bunch_length_m = (
            self.length_scaler.transform(self.bunch_lengths_m[index])
            if self.normalize_lengths
            else self.bunch_lengths_m[index]
        )

        return (rf_setting, formfactor), bunch_length_m


class ShapeReconstructionDataset(Dataset):
    """
    Dataset for reconstructing the bunch shape from RF settings and the THz spectrum.
    """

    def __init__(
        self,
        path: Union[Path, str],
        normalize_rf: bool = True,
        normalize_formfactors: bool = True,
        normalize_currents: bool = True,
    ) -> None:
        self.normalize_rf = normalize_rf
        self.normalize_formfactors = normalize_formfactors
        self.normalize_currents = normalize_currents

        self.rf_settings, self.formfactors, self.currents = self.load_data(path)

        if self.normalize_rf:
            self.rf_scaler = MinMaxScaler().fit(self.rf_settings)
        if self.normalize_formfactors:
            self.formfactor_scaler = MinMaxScaler().fit(self.formfactors)
        if self.normalize_currents:
            self.current_scaler = MinMaxScaler().fit(self.currents)

    def load_data(
        self, path: Union[Path, str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data saved at `path` from disk and return the parts of it relevant to this
        dataset.
        """
        path = Path(path) if isinstance(path, str) else path

        df = pd.read_pickle(path)
        rf_settings = df[["chirp", "chirpL1", "chirpL2", "curv", "skew"]].values

        ss = np.stack(
            [np.linspace(0, 300 * df.loc[i, "slice_width"], num=300) for i in df.index]
        )
        currents = np.stack(df["slice_I"].values)

        formfactors = np.array(
            [
                current2formfactor(
                    ss, currents, grating="both", n_shots=1, clean=False
                )[1]
                for ss, currents in zip(ss, currents)
            ]
        )

        return rf_settings, formfactors, currents

    def __len__(self) -> int:
        return len(self.currents)

    def __getitem__(self, index: int) -> tuple[tuple[np.ndarray, np.ndarray], float]:
        rf_setting = (
            self.rf_scaler.transform(self.rf_settings[index])
            if self.normalize_rf
            else self.rf_settings[index]
        )
        formfactor = (
            self.formfactor_scaler.transform(self.formfactors[index])
            if self.normalize_formfactors
            else self.formfactors[index]
        )
        current = (
            self.current_scaler.transform(self.currents[index])
            if self.normalize_currents
            else self.currents[index]
        )

        return (rf_setting, formfactor), current


class ANNCurrentReconstructor(pl.LightningModule):
    """Neural networks for reconstructing currents at EuXFEL."""

    def __init__(
        self,
        rf_ext_layers: list[int] = [24, 24],
        thz_ext_layers: list[int] = [24, 48],
        encoder_layers: list[int] = [24],
        latent_dim: int = 12,
        decoder_layers: list[int] = [24, 48],
    ) -> None:
        super().__init__()

        self.rf_extractor = nn.Sequential(
            nn.Linear(5, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
        )
        self.thz_extractor = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=10),
            nn.ReLU(),
            nn.Conv1d(24, 48, kernel_size=10),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.encoder = nn.Sequential(
            nn.Linear(100, 24), nn.ReLU(), nn.Linear(24, 12), nn.ReLU()
        )

        self.length_decoder = nn.Sequential()
        self.current_decoder = nn.Sequential()

    def forward(self, rfs, formfactors):
        rf_features = self.rf_extractor(rfs)
        formfactor_features = self.thz_extractor(formfactors)
        features = torch.concat(rf_features, formfactor_features)
        latent = self.encoder(features)
        length = self.length_decoder(latent)
        current = self.current_decoder(latent)
        return length, current
