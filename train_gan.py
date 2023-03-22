# Generator outputs bunch length and shape
# Discriminator is given bunch length and shape of real and generated example and has to
# learn to tell them apart as real or generated

from math import ceil

import torch
from torch import nn


class SignalEncoder(nn.Module):
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


class SignalDecoder(nn.Module):
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

        # output_padding = (output_size - (input_size - 1) * 2 - 1) / 2
        #                  output_size / 2 - (input_size - 1) - 1 / 2
        #                  output_size / 2 - input_size + 1 - 0.5
        #                  output_size / 2 - input_size + 1.5

        # input_sizes = [ceil(self.signal_dims / x) for x in [8, 4, 2]]
        # print(f"{input_sizes = }")
        # output_paddings = [input_size * 2 - () for input_size in input_sizes]
        # output_paddings = [int(output_padding) for output_padding in output_paddings]
        # print(f"{output_paddings = }")

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
        signal = torch.squeeze(x)
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

        self.formfactor_encoder = SignalEncoder(signal_dims=240, latent_dims=10)
        self.current_decoder = SignalDecoder(latent_dims=10 + 5 + 1, signal_dims=300)

    def forward(self, formfactor, rf_settings, bunch_length):
        encoded_formfactor = self.formfactor_encoder(formfactor)
        latent = torch.concatenate(
            [encoded_formfactor, bunch_length, rf_settings], dim=1
        )
        current = self.current_decoder(latent)
        return current


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

        self.formfactor_encoder = SignalEncoder(signal_dims=240, latent_dims=10)
        self.current_encoder = SignalEncoder(signal_dims=300, latent_dims=10)

        self.classifier = nn.Sequential(
            nn.Linear(10 + 10 + 5 + 1, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 1),
        )

    def forward(self, current, formfactor, rf_settings, bunch_length):
        encoded_current = self.current_encoder(current)
        encoded_formfactor = self.formfactor_encoder(formfactor)
        x = torch.concatenate(
            [encoded_current, encoded_formfactor, bunch_length, rf_settings], dim=1
        )
        x = self.classifier(x)
        return x
