from typing import Optional

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
        num_hidden_layers: int = 3,
        hidden_layer_width: int = 100,
        hidden_activation: str = "relu",
        hidden_activation_args: dict = {},
        batch_normalization: bool = True,
    ):
        super().__init__()

        input_dims = rf_settings + formfactor_samples
        self.input_layer = self.hidden_block(
            input_dims,
            hidden_layer_width,
            activation=hidden_activation,
            activation_args=hidden_activation_args,
        )

        blocks = [
            self.hidden_block(
                in_features=hidden_layer_width,
                out_features=hidden_layer_width,
                activation=hidden_activation,
                activation_args=hidden_activation_args,
                batch_normalization=batch_normalization,
                bias=not batch_normalization,
            )
            for _ in range(num_hidden_layers - 1)
        ]
        self.hidden_net = nn.Sequential(*blocks)

        self.current_profile_layer = nn.Sequential(
            nn.Linear(hidden_layer_width, current_samples), nn.Softplus()
        )
        self.bunch_length_layer = nn.Sequential(
            nn.Linear(hidden_layer_width, 1), nn.Softplus()
        )

    def hidden_block(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Optional[str] = None,
        activation_args: dict = {},
        batch_normalization: bool = False,
    ):
        """
        Create a block of a linear layer and an activation, meant to be used as a hidden
        layer in this architecture.
        """
        if activation == "relu":
            activation_module = nn.ReLU(**activation_args)
        elif activation == "leakyrelu":
            activation_module = nn.LeakyReLU(**activation_args)
        elif activation == "softplus":
            activation_module = nn.Softplus(**activation_args)
        elif activation == "sigmoid":
            activation_module = nn.Sigmoid(**activation_args)
        elif activation == "tanh":
            activation_module = nn.Tanh(**activation_args)
        else:
            activation_module = nn.Identity()

        return nn.Sequential(
            nn.Linear(in_features, out_features, bias),
            nn.BatchNorm1d(out_features) if batch_normalization else nn.Identity(),
            activation_module,
        )

    def forward(self, rf_settings, formfactor):
        x = torch.concatenate([rf_settings, formfactor], dim=1)
        x = self.input_layer(x)
        x = self.hidden_net(x)
        current_profile = self.current_profile_layer(x)
        bunch_length = self.bunch_length_layer(x)
        return current_profile, bunch_length


class SupervisedCurrentProfileInference(LightningModule):
    """Model with supervised training for infering current profile at EuXFEL."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        num_hidden_layers: int = 3,
        hidden_layer_width: int = 100,
        hidden_activation: str = "relu",
        hidden_activation_args: dict = {},
        batch_normalization: bool = True,
    ):
        super().__init__()

        self.learning_rate = learning_rate

        self.save_hyperparameters()
        self.example_input_array = [torch.rand(1, 5), torch.rand(1, 240)]

        self.net = MLPCurrentPredictor(
            num_hidden_layers=num_hidden_layers,
            hidden_layer_width=hidden_layer_width,
            hidden_activation=hidden_activation,
            hidden_activation_args=hidden_activation_args,
            batch_normalization=batch_normalization,
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
