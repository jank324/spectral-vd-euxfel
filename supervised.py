from math import ceil
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import LightningModule
from torch import nn, optim

import wandb


class LinearBlock(nn.Module):
    """
    Block with linear layer, optional batch normalisation, and Leaky ReLU activation.
    """

    def __init__(
        self, in_features: int, out_features: int, negative_slope: float = 0.01
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(in_features, out_features, bias=False),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(negative_slope),
        )

    def forward(self, x):
        return self.block(x)


class ConvBlock1d(nn.Module):
    """
    Block with convolutional layer, optional batch normalisation, and Leaky ReLU
    activation.
    """

    def __init__(
        self, in_channels: int, out_channels: int, negative_slope: float = 0.01
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope),
        )

    def forward(self, x):
        return self.block(x)


class ConvEncoder1d(nn.Module):
    """Encodes a 1-dimensional singal using convolutions."""

    def __init__(self, feature_maps: int = 8, negative_slope: float = 0.01):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock1d(1, feature_maps, negative_slope),
            ConvBlock1d(feature_maps, feature_maps * 2, negative_slope),
            ConvBlock1d(feature_maps * 2, 1, negative_slope),
            nn.Flatten(),
        )

    def forward(self, decoded):
        x = torch.unsqueeze(decoded, dim=1)
        encoded = self.encoder(x)
        return encoded


class ConvTransposeBlock1d(nn.Module):
    """
    Block with transposed convolutional layer, optional batch normalisation, and Leaky
    ReLU activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_padding: int = 0,
        negative_slope: float = 0.01,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=output_padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope),
        )

    def forward(self, x):
        return self.block(x)


class ConvDecoder1d(nn.Module):
    """Decodes a 1-dimensional singal using transposed convolutions."""

    def __init__(
        self,
        decoded_dims,
        feature_maps: int = 8,
        negative_slope: float = 0.01,
        output_activation: Optional[nn.Module] = None,
    ):
        super().__init__()

        if output_activation is None:
            output_activation = nn.Identity()

        self.decoder = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(1, ceil(decoded_dims / 8))),
            ConvTransposeBlock1d(
                in_channels=1,
                out_channels=ceil(feature_maps / 2),
                output_padding=(decoded_dims % 8 == 0) * 1,
                negative_slope=negative_slope,
            ),
            ConvTransposeBlock1d(
                in_channels=ceil(feature_maps / 2),
                out_channels=feature_maps,
                output_padding=(decoded_dims % 4 == 0) * 1,
                negative_slope=negative_slope,
            ),
            nn.ConvTranspose1d(
                in_channels=feature_maps,
                out_channels=1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(decoded_dims % 2 == 0) * 1,
            ),
            output_activation,
        )

    def forward(self, encoded):
        x = self.decoder(encoded)
        decoded = torch.squeeze(x, dim=1)
        return decoded


class CurrentProfilePredictor(nn.Module):
    """
    Takes as input a formfactor and the range over which we want the longitudinal
    current profile to be inferred. Outputs a longitudinal current profile.

    Convolve 1-dimensionally the formfactor, maybe do some MLP on the range. Flatten the
    formfactor and concatenate with the output comming from the range. Maybe do some
    more MLP and then deconvolve 1-dimensionally to get the longitudinal current
    profile.
    """

    def __init__(
        self,
        rf_settings: int = 5,
        formfactor_samples: int = 240,
        current_samples: int = 300,
        negative_slope: float = 0.01,
        formfactor_feature_maps: int = 8,
        mlp_width: int = 64,
        current_feature_maps: int = 8,
        softplus_beta: int = 1,
    ) -> None:
        super().__init__()

        self.formfactor_encoder = ConvEncoder1d(formfactor_feature_maps, negative_slope)

        self.mlp = nn.Sequential(
            LinearBlock(
                rf_settings + ceil(formfactor_samples / 8), mlp_width, negative_slope
            ),
            LinearBlock(mlp_width, mlp_width, negative_slope),
            LinearBlock(mlp_width, ceil(current_samples / 8), negative_slope),
        )

        self.current_decoder = ConvDecoder1d(
            300, current_feature_maps, negative_slope, output_activation=nn.ReLU()
        )

        self.bunch_length_decoder = nn.Sequential(
            nn.Linear(ceil(current_samples / 8), 1), nn.Softplus(softplus_beta)
        )

    def forward(self, rf_settings, formfactor):
        x = self.formfactor_encoder(formfactor)
        x = torch.concatenate([rf_settings, x], dim=1)
        x = self.mlp(x)
        current_profile = self.current_decoder(x)
        bunch_length = self.bunch_length_decoder(x)
        return current_profile, bunch_length


class SupervisedCurrentProfileInference(LightningModule):
    """Model with supervised training for infering current profile at EuXFEL."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        mlp_width: int = 64,
        formfactor_feature_maps: int = 8,
        current_feature_maps: int = 8,
        negative_slope: float = 0.01,
        softplus_beta: int = 1,
    ):
        super().__init__()

        self.learning_rate = learning_rate

        self.save_hyperparameters()
        self.example_input_array = [torch.rand(1, 5), torch.rand(1, 240)]

        self.net = CurrentProfilePredictor(
            mlp_width=mlp_width,
            formfactor_feature_maps=formfactor_feature_maps,
            current_feature_maps=current_feature_maps,
            negative_slope=negative_slope,
            softplus_beta=softplus_beta,
        )

        self.current_criterion = nn.MSELoss()
        self.length_criterion = nn.MSELoss()

    def configure_optimizers(self):
        return optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def forward(self, rf_settings, formfactor):
        current_profile, bunch_length = self.net(rf_settings, formfactor)
        return current_profile, bunch_length

    def training_step(self, batch, batch_idx):
        (rf_settings, formfactors), (true_currents, true_lengths) = batch

        predicted_currents, predicted_lengths = self.net(rf_settings, formfactors)

        current_loss = self.current_criterion(predicted_currents, true_currents)
        length_loss = self.length_criterion(predicted_lengths, true_lengths)
        loss = current_loss + length_loss

        self.log("train/current_loss", current_loss)
        self.log("train/length_loss", length_loss)
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        (rf_settings, formfactors), (true_currents, true_lengths) = batch

        predicted_currents, predicted_lengths = self.net(rf_settings, formfactors)

        current_loss = self.current_criterion(predicted_currents, true_currents)
        length_loss = self.length_criterion(predicted_lengths, true_lengths)
        loss = current_loss + length_loss

        self.log("validate/current_loss", current_loss)
        self.log("validate/length_loss", length_loss)
        self.log("validate/loss", loss)

        if batch_idx == 0:
            self.log_current_profile_sample_plot(
                true_currents, true_lengths, predicted_currents, predicted_lengths
            )

        return loss

    def test_step(self, batch, batch_idx):
        (rf_settings, formfactors), (true_currents, true_lengths) = batch

        predicted_currents, predicted_lengths = self.net(rf_settings, formfactors)

        current_loss = self.current_criterion(predicted_currents, true_currents)
        length_loss = self.length_criterion(predicted_lengths, true_lengths)
        loss = current_loss + length_loss

        self.log("test/current_loss", current_loss)
        self.log("test/length_loss", length_loss)
        self.log("test/loss", loss)

        return loss

    def log_current_profile_sample_plot(
        self,
        real_current_profile_batch,
        real_bunch_length_batch,
        fake_current_profile_batch,
        fake_bunch_length_batch,
    ):
        """
        Logs a plot comparing the real current profile to the generated one to
        Weights & Biases.
        """
        real_current_profile_batch = real_current_profile_batch.cpu().detach().numpy()
        real_bunch_length_batch = real_bunch_length_batch.cpu().detach().numpy()
        fake_current_profile_batch = fake_current_profile_batch.cpu().detach().numpy()
        fake_bunch_length_batch = fake_bunch_length_batch.cpu().detach().numpy()

        fig, axs = plt.subplots(2, 4)
        axs = axs.flatten()
        for i in range(8):
            ss_real = np.linspace(
                -real_bunch_length_batch[i][0] / 2,
                real_bunch_length_batch[i][0] / 2,
                num=300,
            )
            axs[i].plot(ss_real, real_current_profile_batch[i], label="Real")

            ss_fake = np.linspace(
                -fake_bunch_length_batch[i][0] / 2,
                fake_bunch_length_batch[i][0] / 2,
                num=300,
            )
            axs[i].plot(ss_fake, fake_current_profile_batch[i], label="Fake")

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels)

        wandb.log({"real_vs_fake_validation_plot": fig})
