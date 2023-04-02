from math import ceil

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torchmetrics.functional import mean_absolute_error

import wandb


class ConvolutionalEncoder(nn.Module):
    """Encodes a signal to a latent vector."""

    def __init__(
        self,
        signal_dims: int,
        latent_dims: int,
        norm: nn.Module = nn.BatchNorm1d,
        leaky_relu_negative_slope: float = 0.2,
    ) -> None:
        super().__init__()

        self.convnet = nn.Sequential(
            nn.Conv1d(1, 8, 3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(leaky_relu_negative_slope),
            nn.Conv1d(8, 16, 3, stride=2, padding=1, bias=False),
            norm(16, affine=True),
            nn.LeakyReLU(leaky_relu_negative_slope),
            nn.Conv1d(16, 1, 3, stride=2, padding=1, bias=False),
            norm(1, affine=True),
            nn.LeakyReLU(leaky_relu_negative_slope),
        )

        self.flatten = nn.Flatten()

        self.linear = nn.Sequential(
            nn.Linear(ceil(signal_dims / 8) * 1, latent_dims),
            nn.LeakyReLU(leaky_relu_negative_slope),
        )

    def forward(self, signal):
        x = torch.unsqueeze(signal, dim=1)
        x = self.convnet(x)
        x = self.flatten(x)
        encoded = self.linear(x)
        return encoded


class ConvolutionalDecoder(nn.Module):
    """
    Decodes a signal from a latent vector.

    NOTE: Can currently only decode non-negative signals.
    """

    def __init__(
        self, latent_dims: int, signal_dims: int, leaky_relu_negative_slope: float = 0.2
    ) -> None:
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(latent_dims, ceil(signal_dims / 8) * 1),
            nn.LeakyReLU(leaky_relu_negative_slope),
        )

        self.unflatten = nn.Unflatten(
            dim=1, unflattened_size=(1, ceil(signal_dims / 8))
        )

        self.convnet = nn.Sequential(
            nn.ConvTranspose1d(
                1,
                16,
                3,
                stride=2,
                padding=1,
                output_padding=(signal_dims % 8 == 0) * 1,
                bias=False,
            ),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(leaky_relu_negative_slope),
            nn.ConvTranspose1d(
                16,
                8,
                3,
                stride=2,
                padding=1,
                output_padding=(signal_dims % 4 == 0) * 1,
                bias=False,
            ),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(leaky_relu_negative_slope),
            nn.ConvTranspose1d(
                8,
                1,
                3,
                stride=2,
                padding=1,
                output_padding=(signal_dims % 2 == 0) * 1,
            ),
            nn.ReLU(),
        )

    def forward(self, encoded):
        x = self.linear(encoded)
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

    def __init__(
        self,
        num_rf_settings: int = 5,
        num_formfactor_samples: int = 240,
        num_current_samples: int = 300,
        encoded_formfactor_dims: int = 10,
        latent_dims: int = 10,
        leaky_relu_negative_slope: float = 0.2,
    ) -> None:
        super().__init__()

        self.formfactor_encoder = ConvolutionalEncoder(
            signal_dims=num_formfactor_samples,
            latent_dims=encoded_formfactor_dims,
            norm=nn.BatchNorm1d,
            leaky_relu_negative_slope=leaky_relu_negative_slope,
        )
        self.scalar_spectral_combine_mlp = nn.Sequential(
            nn.Linear(encoded_formfactor_dims + num_rf_settings, 50),
            nn.LeakyReLU(leaky_relu_negative_slope),
            nn.Linear(50, 20),
            nn.LeakyReLU(leaky_relu_negative_slope),
            nn.Linear(20, latent_dims),
            nn.LeakyReLU(leaky_relu_negative_slope),
        )
        self.current_decoder = ConvolutionalDecoder(
            latent_dims=latent_dims,
            signal_dims=num_current_samples,
            leaky_relu_negative_slope=leaky_relu_negative_slope,
        )
        self.bunch_length_decoder = nn.Sequential(
            nn.Linear(latent_dims, 20),
            nn.LeakyReLU(leaky_relu_negative_slope),
            nn.Linear(20, 1),
            nn.Softplus(),
        )

    def forward(self, rf_settings, formfactor):
        encoded_formfactor = self.formfactor_encoder(formfactor)
        x = torch.concatenate([rf_settings, encoded_formfactor], dim=1)
        latent = self.scalar_spectral_combine_mlp(x)
        current_profile = self.current_decoder(latent)
        bunch_length = self.bunch_length_decoder(latent)
        return current_profile, bunch_length


class Critic(nn.Module):
    """
    Take as input a longitudinal current profile and the range over which that current
    profile is given. Output value between 1 and 0 on whether the input example was a
    real measured one or a prediction from the Generator.

    Convolve the current profile 1-dimensionally, then flatten. Maybe do some MLP on the
    range. Concateneate decoded current profile and range, then do more MLP down to the
    1-element output.
    """

    def __init__(
        self,
        num_rf_settings: int = 5,
        num_formfactor_samples: int = 240,
        num_current_samples: int = 300,
        encoded_formfactor_dims: int = 10,
        encoded_current_dims: int = 10,
        leaky_relu_negative_slope: float = 0.2,
    ) -> None:
        super().__init__()

        self.formfactor_encoder = ConvolutionalEncoder(
            signal_dims=num_formfactor_samples,
            latent_dims=encoded_formfactor_dims,
            norm=nn.InstanceNorm1d,
            leaky_relu_negative_slope=leaky_relu_negative_slope,
        )
        self.current_encoder = ConvolutionalEncoder(
            signal_dims=num_current_samples,
            latent_dims=encoded_current_dims,
            norm=nn.InstanceNorm1d,
            leaky_relu_negative_slope=leaky_relu_negative_slope,
        )

        classifier_input_dims = (
            num_rf_settings + encoded_formfactor_dims + encoded_current_dims + 1
        )
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dims, 50),
            nn.LeakyReLU(leaky_relu_negative_slope),
            nn.Linear(50, 20),
            nn.LeakyReLU(leaky_relu_negative_slope),
            nn.Linear(20, 1),
        )

    def forward(self, rf_settings, formfactor, current_profile, bunch_length):
        encoded_current_profile = self.current_encoder(current_profile)
        encoded_formfactor = self.formfactor_encoder(formfactor)
        x = torch.concatenate(
            [rf_settings, encoded_formfactor, encoded_current_profile, bunch_length],
            dim=1,
        )
        x = self.classifier(x)
        return x


def initialize_weights(model):
    """
    Initialise the weights of `Conv1d`, `ConvTranspose1d` and `BatchNorm1d` layers in
    `model` according to DCGAN paper.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


class WassersteinGANGP(L.LightningModule):
    """Wasserstein GAN with Gradient Penalty for infering current profile at EuXFEL."""

    def __init__(
        self,
        critic_iterations: int = 5,
        lambda_gradient_penalty: float = 10.0,
        learning_rate=1e-4,
        leaky_relu_negative_slope: float = 0.2,
    ):
        super().__init__()

        self.critic_iterations = critic_iterations
        self.lambda_gradient_penalty = lambda_gradient_penalty
        self.learning_rate = learning_rate

        self.save_hyperparameters()
        self.automatic_optimization = False
        self.example_input_array = [torch.rand(1, 5), torch.rand(1, 240)]

        self.generator = Generator(leaky_relu_negative_slope=leaky_relu_negative_slope)
        self.critic = Critic(leaky_relu_negative_slope=leaky_relu_negative_slope)

        initialize_weights(self.generator)
        initialize_weights(self.critic)

    def forward(self, rf_settings, formfactor):
        current_profile, bunch_length = self.generator(rf_settings, formfactor)
        return current_profile, bunch_length

    def configure_optimizers(self):
        generator_optimizer = optim.Adam(
            self.generator.parameters(), lr=self.learning_rate, betas=(0.0, 0.9)
        )
        critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.learning_rate, betas=(0.0, 0.9)
        )
        return generator_optimizer, critic_optimizer

    def gradient_penalty_loss(
        self,
        real_current_profiles,
        fake_current_profiles,
        real_bunch_lengths,
        fake_bunch_lengths,
        formfactors,
        rf_settings,
    ):
        # Interpolate real and generated current profiles
        batch_size, num_current_samples = real_current_profiles.size()
        alpha_1 = (
            torch.rand(batch_size, 1)
            .repeat(1, num_current_samples)
            .type_as(fake_current_profiles)
        )
        interpolated_current_profiles = (
            alpha_1 * real_current_profiles + (1 - alpha_1) * fake_current_profiles
        )
        alpha_2 = torch.rand(batch_size, 1).type_as(fake_bunch_lengths)
        interpolated_bunch_lengths = (
            alpha_2 * real_bunch_lengths + (1 - alpha_2) * fake_bunch_lengths
        )

        # Calculate critic scores
        mixed_critiques = self.critic(
            rf_settings,
            formfactors,
            interpolated_current_profiles,
            interpolated_bunch_lengths,
        )

        # Take the gradient of the critic outputs with respect to the current profiles
        gradient = torch.autograd.grad(
            outputs=mixed_critiques,
            inputs=[interpolated_current_profiles, interpolated_bunch_lengths],
            grad_outputs=torch.ones_like(mixed_critiques),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty

    def training_step(self, batch, batch_idx):
        (rf_settings, formfactors), (real_current_profiles, real_bunch_lengths) = batch

        generator_optimizer, critic_optimizer = self.optimizers()

        # Train critic
        for _ in range(self.critic_iterations):
            fake_current_profiles, fake_bunch_lengths = self.generator(
                rf_settings, formfactors
            )
            critique_real = self.critic(
                rf_settings, formfactors, real_current_profiles, real_bunch_lengths
            )
            critique_fake = self.critic(
                rf_settings,
                formfactors,
                fake_current_profiles,
                fake_bunch_lengths,
            )
            wasserstein_loss = -(torch.mean(critique_real) - torch.mean(critique_fake))
            gradient_penalty = self.gradient_penalty_loss(
                real_current_profiles,
                fake_current_profiles,
                real_bunch_lengths,
                fake_bunch_lengths,
                formfactors,
                rf_settings,
            )
            critic_loss = (
                wasserstein_loss + self.lambda_gradient_penalty * gradient_penalty
            )
            critic_optimizer.zero_grad()
            self.manual_backward(critic_loss)
            critic_optimizer.step()
        self.log("train/wasserstein_loss", wasserstein_loss)
        self.log("train/gradient_penalty", gradient_penalty)
        self.log("train/critic_loss", critic_loss)

        # Train generator
        fake_current_profiles, fake_bunch_lengths = self.generator(
            rf_settings, formfactors
        )
        critique_fake = self.critic(
            rf_settings,
            formfactors,
            fake_current_profiles,
            fake_bunch_lengths,
        )
        generator_loss = -torch.mean(critique_fake)
        self.log("train/generator_loss", generator_loss)
        self.log(
            "train/bunch_length_mean_absolute_error",
            mean_absolute_error(fake_bunch_lengths, real_bunch_lengths),
        )
        generator_optimizer.zero_grad()
        self.manual_backward(generator_loss)
        generator_optimizer.step()

    def validation_step(self, batch, batch_idx):
        (rf_settings, formfactors), (real_current_profiles, real_bunch_lengths) = batch

        fake_current_profiles, fake_bunch_lengths = self.generator(
            rf_settings, formfactors
        )
        critique_real = self.critic(
            rf_settings, formfactors, real_current_profiles, real_bunch_lengths
        )
        critique_fake = self.critic(
            rf_settings, formfactors, fake_current_profiles, fake_bunch_lengths
        )
        wasserstein_loss = -(torch.mean(critique_real) - torch.mean(critique_fake))
        generator_loss = -torch.mean(critique_fake)

        self.log("validate/wasserstein_loss", wasserstein_loss, sync_dist=True)
        self.log("validate/generator_loss", generator_loss, sync_dist=True)
        self.log(
            "validate/bunch_length_mean_absolute_error",
            mean_absolute_error(fake_bunch_lengths, real_bunch_lengths),
        )

        if batch_idx == 0:
            self.log_current_profile_sample_plot(
                real_current_profiles,
                real_bunch_lengths,
                fake_current_profiles,
                fake_bunch_lengths,
            )

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
