from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split

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


class LengthReconstructor(pl.LightningModule):
    """Neural networks for reconstructing currents at EuXFEL."""

    def __init__(self) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(5 + 300, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.ReLU(),
        )

    def forward(
        self, rf_settings: torch.Tensor, formfactors: torch.Tensor
    ) -> torch.Tensor:
        concatenated = torch.concat([rf_settings, formfactors])
        bunch_length = self.mlp(concatenated)
        return bunch_length

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, train_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        (rf_settings, formfactors), bunch_lengths = train_batch
        predictions = self.forward(rf_settings, formfactors)
        loss = F.mse_loss(predictions, bunch_lengths)
        self.log("train_loss", loss)
        mae = F.l1_loss(predictions, bunch_lengths)
        self.log("train_mae", mae)
        return loss

    def validation_step(self, val_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        (rf_settings, formfactors), bunch_lengths = val_batch
        predictions = self.forward(rf_settings, formfactors)
        loss = F.mse_loss(predictions, bunch_lengths)
        self.log("train_loss", loss)
        mae = F.l1_loss(predictions, bunch_lengths)
        self.log("train_mae", mae)
        return loss


def main() -> None:
    dataset = LengthReconstructionDataset("data/zihan/data_20220905.pkl")
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    model = LengthReconstructor()

    trainer = pl.Trainer()
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
