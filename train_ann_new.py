from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import Dataset

from .utils import current2formfactor


class CurrentReconstructionDataset(Dataset):
    """Dataset for training a neural network on current reconstruction at EuXFEL."""

    def __init__(self, path: Union[Path, str]) -> None:
        path = Path(path) if isinstance(path, str) else path

        df = pd.read_pickle(path)

        self.rfs_dataset = df[["chirp", "chirpL1", "chirpL2", "curv", "skew"]].values

        ss_dataset = np.stack(
            [np.linspace(0, 300 * df.loc[i, "slice_width"], num=300) for i in df.index]
        )
        self.lengths = ss_dataset.max(axis=1) - ss_dataset.min()

        self.currents_dataset = np.stack(df["slice_I"].values)

        self.formfactors_dataset = np.array(
            [
                current2formfactor(
                    ss, currents, grating="both", n_shots=1, clean=False
                )[1]
                for ss, currents in zip(ss_dataset, self.currents_dataset)
            ]
        )

    def __len__(self) -> int:
        return len(self.rfs_dataset)

    def __getitem__(self, index: int):
        X = (self.rfs_dataset[index], self.formfactors_dataset[index])
        y = np.concatenate([[self.lengths[index]], self.currents_dataset[index]])
        return X, y


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
