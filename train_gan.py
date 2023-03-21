# Generator outputs bunch length and shape
# Discriminator is given bunch length and shape of real and generated example and has to
# learn to tell them apart as real or generated

import torch
from torch import nn


class FormfactorEncoder(nn.Module):
    """Encodes formfactors to a latent vector."""

    def __init__(self) -> None:
        super().__init__()

        self.convnet = nn.Sequential(
            nn.Conv1d(1, 8, 3, stride=2, padding=1),  # 120
            nn.LeakyReLU(),
            nn.Conv1d(8, 16, 3, stride=2, padding=1),  # 60
            nn.LeakyReLU(),
            nn.Conv1d(16, 32, 3, stride=2, padding=1),  # 30
            nn.LeakyReLU(),
        )

        self.flatten = nn.Flatten()

        self.mlp = nn.Sequential(
            nn.Linear(30 * 32, 100), nn.LeakyReLU(), nn.Linear(100, 10)
        )

    def forward(self, formfactor):
        x = self.convnet(formfactor)
        x = self.flatten(x)
        encoded = self.mlp(x)
        return encoded


class CurrentEncoder(nn.Module):
    """Encodes a current profile to a latent vector."""

    def __init__(self) -> None:
        super().__init__()

        self.convnet = nn.Sequential(
            nn.Conv1d(1, 8, 3, stride=2, padding=1),  # 120
            nn.LeakyReLU(),
            nn.Conv1d(8, 16, 3, stride=2, padding=1),  # 60
            nn.LeakyReLU(),
            nn.Conv1d(16, 32, 3, stride=2, padding=1),  # 30
            nn.LeakyReLU(),
        )

        self.flatten = nn.Flatten()

        self.mlp = nn.Sequential(
            nn.Linear(30 * 32, 100), nn.LeakyReLU(), nn.Linear(100, 10)
        )

    def forward(self, current):
        x = self.convnet(current)
        x = self.flatten(x)
        encoded = self.mlp(x)
        return encoded


class CurrentDecoder(nn.Module):
    """Decodes a current profile from a latent vector."""

    def __init__(self) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(10 + 5 + 1, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 100),
            nn.LeakyReLU(),
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 37.5))

        self.convnet = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 75
            nn.LeakyReLU(),
            nn.ConvTranspose1d(16, 8, 3, stride=2, padding=1, output_padding=1),  # 150
            nn.LeakyReLU(),
            nn.ConvTranspose1d(8, 1, 3, stride=2, padding=1, output_padding=1),  # 300
            nn.ReLU(),
        )

    def forward(self, encoded):
        x = self.current_decoder_mlp(encoded)
        x = self.unflatten(x)
        current = self.current_decoder_conv(x)
        return current


class Generator(nn.Module):
    """
    Takes as input a formfactor and the range over which we want the longitudinal
    current profile to be inferred. Outputs a longitudinal current profile.

    Convolve 1-dimensionally the formfactor, maybe do some MLP on the range. Flatten the
    formfactor and concatenate with the output comming from the range. Maybe do some
    more MLP and then deconvolve 1-dimensionally to get the longitudinal current
    profile.
    """

    def __init__(self, formfactor_encoder, current_decoder) -> None:
        super().__init__()

        self.formfactor_encoder = formfactor_encoder
        self.current_decoder = current_decoder

    def forward(self, formfactor, rf_settings, bunch_length):
        encoded_formfactor = self.formfactor_encoder(formfactor)
        latent = torch.concatenate([encoded_formfactor, bunch_length, rf_settings])
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

    def __init__(self, formfactor_encoder, current_encoder) -> None:
        super().__init__()

        self.formfactor_encoder = formfactor_encoder
        self.current_encoder = current_encoder

        self.classifier = nn.Sequential(
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 1),
        )

    def forward(self, current, formfactor, rf_settings, bunch_length):
        encoded_current = self.current_encoder(current)
        encoded_formfactor = self.formfactor_encoder(formfactor)
        x = torch.concatenate(
            [encoded_current, encoded_formfactor, bunch_length, rf_settings]
        )
        return self.classifier(x)
