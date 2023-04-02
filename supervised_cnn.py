import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

import wandb
from gan import Generator


class CNNCurrentReconstructor(L.LightningModule):
    """
    Convolutional model with supervised training for infering current profile at EuXFEL.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        leaky_relu_negative_slope: float = 0.01,
    ):
        super().__init__()

        self.learning_rate = learning_rate

        self.save_hyperparameters()
        self.example_input_array = [torch.rand(1, 5), torch.rand(1, 240)]

        self.generator = Generator(leaky_relu_negative_slope=leaky_relu_negative_slope)

        self.current_criterion = nn.MSELoss()
        self.length_criterion = nn.MSELoss()

    def configure_optimizers(self):
        return optim.Adam(self.generator.parameters(), lr=self.learning_rate)

    def forward(self, rf_settings, formfactor):
        current_profile, bunch_length = self.generator(rf_settings, formfactor)
        return current_profile, bunch_length

    def training_step(self, batch, batch_idx):
        (rf_settings, formfactors), (real_current_profiles, real_bunch_lengths) = batch

        fake_current_profiles, fake_bunch_lengths = self.generator(
            rf_settings, formfactors
        )

        current_loss = self.current_criterion(
            fake_current_profiles, real_current_profiles
        )
        length_loss = self.length_criterion(fake_bunch_lengths, real_bunch_lengths)
        loss = current_loss + length_loss

        self.log("train/current_loss", current_loss)
        self.log("train/length_loss", length_loss)
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        (rf_settings, formfactors), (real_current_profiles, real_bunch_lengths) = batch

        fake_current_profiles, fake_bunch_lengths = self.generator(
            rf_settings, formfactors
        )

        current_loss = self.current_criterion(
            fake_current_profiles, real_current_profiles
        )
        length_loss = self.length_criterion(fake_bunch_lengths, real_bunch_lengths)
        loss = current_loss + length_loss

        self.log("validate/current_loss", current_loss)
        self.log("validate/length_loss", length_loss)
        self.log("validate/loss", loss)

        if batch_idx == 0:
            self.log_current_profile_sample_plot(
                real_current_profiles,
                real_bunch_lengths,
                fake_current_profiles,
                fake_bunch_lengths,
            )

        return loss

    def test_step(self, batch, batch_idx):
        (rf_settings, formfactors), (real_current_profiles, real_bunch_lengths) = batch

        fake_current_profiles, fake_bunch_lengths = self.generator(
            rf_settings, formfactors
        )

        current_loss = self.current_criterion(
            fake_current_profiles, real_current_profiles
        )
        length_loss = self.length_criterion(fake_bunch_lengths, real_bunch_lengths)
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
