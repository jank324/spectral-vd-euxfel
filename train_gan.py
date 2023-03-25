# Generator outputs bunch length and shape
# Discriminator is given bunch length and shape of real and generated example and has to
# learn to tell them apart as real or generated

# TODO Implement MLP encoder and decoder
# TODO Implement supervised single-model training setup
# TODO Add data normalisation

from math import ceil

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.loggers import WandbLogger
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split

from utils import current2formfactor


class ConvolutionalEncoder(nn.Module):
    """Encodes a signal to a latent vector."""

    def __init__(self, signal_dims, latent_dims) -> None:
        super().__init__()

        self.convnet = nn.Sequential(
            nn.Conv1d(1, 8, 3, stride=2, padding=1),  # 120 / 150
            nn.LeakyReLU(),
            nn.Conv1d(8, 16, 3, stride=2, padding=1),  # 60 / 75
            nn.LeakyReLU(),
            nn.Conv1d(16, 32, 3, stride=2, padding=1),  # 30 / 37.5 ?
            nn.LeakyReLU(),
        )

        self.flatten = nn.Flatten()

        self.mlp = nn.Sequential(
            nn.Linear(ceil(signal_dims / 8) * 32, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.Linear(50, latent_dims),
        )

    def forward(self, signal):
        x = torch.unsqueeze(signal, dim=1)
        x = self.convnet(x)
        x = self.flatten(x)
        encoded = self.mlp(x)
        return encoded


class ConvolutionalDecoder(nn.Module):
    """Decodes a signal from a latent vector."""

    def __init__(self, latent_dims, signal_dims) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(latent_dims, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 100),
            nn.LeakyReLU(),
            nn.Linear(100, ceil(signal_dims / 8) * 32),
            nn.LeakyReLU(),
        )

        self.unflatten = nn.Unflatten(
            dim=1, unflattened_size=(32, ceil(signal_dims / 8))
        )

        self.convnet = nn.Sequential(
            nn.ConvTranspose1d(
                32,
                16,
                3,
                stride=2,
                padding=1,
                output_padding=(signal_dims % 8 == 0) * 1,
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(
                16, 8, 3, stride=2, padding=1, output_padding=(signal_dims % 4 == 0) * 1
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(
                8, 1, 3, stride=2, padding=1, output_padding=(signal_dims % 2 == 0) * 1
            ),
            nn.ReLU(),
        )

    def forward(self, encoded):
        x = self.mlp(encoded)
        x = self.unflatten(x)
        x = self.convnet(x)
        signal = torch.squeeze(x, dim=1)
        return signal


class Generator(nn.Module):
    """
    Takes as input a formfactor and the range over which we want the longitudinal
    current profile to be inferred. Outputs a longitudinal current profile.

    Convolve 1-dimensionally the formfactor, maybe do some MLP on the range. Flatten the
    formfactor and concatenate with the output comming from the range. Maybe do some
    more MLP and then deconvolve 1-dimensionally to get the longitudinal current
    profile.
    """

    def __init__(self) -> None:
        super().__init__()

        self.formfactor_encoder = ConvolutionalEncoder(signal_dims=240, latent_dims=10)
        self.current_decoder = ConvolutionalDecoder(
            latent_dims=10 + 5 + 1, signal_dims=300
        )

    def forward(self, formfactor, rf_settings, bunch_length):
        encoded_formfactor = self.formfactor_encoder(formfactor)
        latent = torch.concatenate(
            [encoded_formfactor, bunch_length, rf_settings], dim=1
        )
        current_profile = self.current_decoder(latent)
        return current_profile


class Critic(nn.Module):
    """
    Take as input a longitudinal current profile and the range over which that current
    profile is given. Output value between 1 and 0 on whether the input example was a
    real measured one or a prediction from the Generator.

    Convolve the current profile 1-dimensionally, then flatten. Maybe do some MLP on the
    range. Concateneate decoded current profile and range, then do more MLP down to the
    1-element output.
    """

    def __init__(self) -> None:
        super().__init__()

        self.formfactor_encoder = ConvolutionalEncoder(signal_dims=240, latent_dims=10)
        self.current_encoder = ConvolutionalEncoder(signal_dims=300, latent_dims=10)

        self.classifier = nn.Sequential(
            nn.Linear(10 + 10 + 5 + 1, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 1),  # TODO Activation on output?
        )

    def forward(self, current_profile, formfactor, rf_settings, bunch_length):
        encoded_current_profile = self.current_encoder(current_profile)
        encoded_formfactor = self.formfactor_encoder(formfactor)
        x = torch.concatenate(
            [encoded_current_profile, encoded_formfactor, bunch_length, rf_settings],
            dim=1,
        )
        x = self.classifier(x)
        return x


class WassersteinGANGP(L.LightningModule):
    """Wasserstein GAN with Gradient Penalty for infering current profile at EuXFEL."""

    def __init__(
        self, critic_iterations: int = 5, lambda_gradient_penalty: float = 10.0
    ):
        super().__init__()

        self.critic_iterations = critic_iterations
        self.lambda_gradient_penalty = lambda_gradient_penalty

        self.save_hyperparameters()
        self.automatic_optimization = False
        self.example_input_array = [
            torch.rand(1, 240),
            torch.rand(1, 5),
            torch.rand(1, 1),
        ]

        self.generator = Generator()
        self.critic = Critic()

    def forward(self, formfactor, rf_settings, bunch_length):
        current_profile = self.generator(formfactor, rf_settings, bunch_length)
        return current_profile

    def configure_optimizers(self):
        generator_optimizer = optim.Adam(self.generator.parameters(), lr=1e-3)
        critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        return generator_optimizer, critic_optimizer

    def gradient_penalty_loss(
        self,
        real_current_profiles,
        generated_current_profiles,
        formfactors,
        rf_settings,
        bunch_lengths,
    ):
        # Interpolate real and generated current profiles
        batch_size, n_current_samples = real_current_profiles.size()
        alpha = (
            torch.rand(batch_size, 1)
            .repeat(1, n_current_samples)
            .type_as(generated_current_profiles)
        )
        interpolated_current_profiles = (
            alpha * real_current_profiles + (1 - alpha) * generated_current_profiles
        )

        # Calculate critic scores
        mixed_critiques = self.critic(
            interpolated_current_profiles, formfactors, rf_settings, bunch_lengths
        )

        # Take the gradient of the critic outputs with respect to the current profiles
        gradient = torch.autograd.grad(
            outputs=mixed_critiques,
            inputs=interpolated_current_profiles,
            grad_outputs=torch.ones_like(mixed_critiques),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty

    def training_step(self, batch, batch_idx):
        (formfactors, rf_settings, bunch_lengths), real_current_profiles = batch

        generator_optimizer, critic_optimizer = self.optimizers()

        # Train critic
        for _ in range(self.critic_iterations):
            generated_current_profiles = self.generator(
                formfactors, rf_settings, bunch_lengths
            )
            critique_real = self.critic(
                real_current_profiles, formfactors, rf_settings, bunch_lengths
            )
            critique_fake = self.critic(
                generated_current_profiles, formfactors, rf_settings, bunch_lengths
            )
            gradient_penalty = self.gradient_penalty_loss(
                real_current_profiles,
                generated_current_profiles,
                formfactors,
                rf_settings,
                bunch_lengths,
            )
            critic_loss = (
                -(torch.mean(critique_real) - torch.mean(critique_fake))
                + self.lambda_gradient_penalty * gradient_penalty
            )
            critic_optimizer.zero_grad()
            self.manual_backward(critic_loss)
            critic_optimizer.step()
        self.log("train/critic_loss", critic_loss)

        # Train generator
        generated_current_profiles = self.generator(
            formfactors, rf_settings, bunch_lengths
        )
        # TODO Log generated current profiles
        critique_fake = self.critic(
            generated_current_profiles, formfactors, rf_settings, bunch_lengths
        )
        generator_loss = -torch.mean(critique_fake)
        self.log("train/generator_loss", generator_loss)
        generator_optimizer.zero_grad()
        self.manual_backward(generator_loss)
        generator_optimizer.step()


class EuXFELCurrentDataset(Dataset):
    """
    Dataset on reconstructing current profiles from THz spectrum formfactors and RF
    settings with bunch lengths also given as an input to the prediction.

    `X` is the formfactors, RF settings and bunch lengths. `y` is the current profiles.
    """

    def __init__(self):
        df = pd.read_pickle("data/zihan/data_20220905.pkl")

        self.current_profiles = np.stack(df.loc[:, "slice_I"])
        self.rf_settings = df.loc[
            :, ["chirp", "curv", "skew", "chirpL1", "chirpL2"]
        ].values
        self.bunch_lengths = (
            df.loc[:, "slice_width"].values * self.current_profiles.shape[1]
        )

        self.formfactors = [
            current2formfactor(
                ss=np.linspace(0, bunch_length, num=len(current_profile)),
                currents=np.array(current_profile),
                grating="both",
                clean=False,
                n_shots=10,
            )[1]
            for current_profile, bunch_length in zip(
                self.current_profiles, self.bunch_lengths
            )
        ]

    def __len__(self):
        return len(self.bunch_lengths)

    def __getitem__(self, index):
        formfactor = self.formfactors[index]
        rf_settings = self.rf_settings[index]
        bunch_length = self.bunch_lengths[index]
        current_profile = self.current_profiles[index]

        formfactor = torch.tensor(formfactor, dtype=torch.float32)
        rf_settings = torch.tensor(rf_settings, dtype=torch.float32)
        bunch_length = torch.tensor(bunch_length, dtype=torch.float32).unsqueeze(dim=0)
        current_profile = torch.tensor(current_profile, dtype=torch.float32)

        return (formfactor, rf_settings, bunch_length), current_profile


class EuXFELCurrentDataModule(L.LightningDataModule):
    """
    Data Module for reconstructing current profiles from THz spectrum formfactors and RF
    settings with bunch lengths also given as an input to the prediction.
    """

    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage):
        dataset_full = EuXFELCurrentDataset()

        self.dataset_train, self.dataset_val, self.dataset_test = random_split(
            dataset_full, [0.6, 0.2, 0.2]
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train, batch_size=self.batch_size, num_workers=10, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=10)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=10)

    def predict_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=10)


def main():
    data_module = EuXFELCurrentDataModule(batch_size=32)
    model = WassersteinGANGP()

    wandb_logger = WandbLogger(project="virtual-diagnostics-euxfel-current-gan")

    # TODO Fix errors raised when running on accelerator="mps"
    trainer = L.Trainer(logger=wandb_logger, fast_dev_run=True, accelerator="cpu")
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
