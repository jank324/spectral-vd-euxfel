import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import LightningModule
from torch import nn, optim

import wandb


class MLPCurrentPredictor(nn.Module):
    """
    MLP model for inferring the current profile from the RF settings and the THz
    formfactor.

    This model largely follows the architecture previously used (and working) in
    Tensorflow. The only major difference is that current profile and bunch length
    prediction are now done by a single model instead of two separate ones.
    """

    def __init__(
        self,
        rf_settings: int = 5,
        formfactor_samples: int = 240,
        current_samples: int = 300,
    ):
        super().__init__()

        self.hidden_net = nn.Sequential(
            nn.Linear(rf_settings + formfactor_samples, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
        )

        self.current_profile_layer = nn.Sequential(
            nn.Linear(50, current_samples), nn.Softplus()
        )
        self.bunch_length_layer = nn.Sequential(nn.Linear(50, 1), nn.Softplus())

    def forward(self, rf_settings, formfactor):
        x = torch.concatenate([rf_settings, formfactor], dim=1)
        x = self.hidden_net(x)
        current_profile = self.current_profile_layer(x)
        bunch_length = self.bunch_length_layer(x)
        return current_profile, bunch_length


class SupervisedCurrentProfileInference(LightningModule):
    """Model with supervised training for infering current profile at EuXFEL."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
    ):
        super().__init__()

        self.learning_rate = learning_rate

        self.save_hyperparameters()
        self.example_input_array = [torch.rand(1, 5), torch.rand(1, 240)]

        self.net = MLPCurrentPredictor()

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
