# Generator outputs bunch length and shape
# Discriminator is given bunch length and shape of real and generated example and has to
# learn to tell them apart as real or generated

import torch
from torch import nn


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

        self.formfactor_encoder = nn.Sequential(
            nn.Conv1d(), nn.Conv1d(), nn.Conv1d(), nn.Flatten()
        )
        self.current_decoder = nn.Sequential(
            nn.Conv1d()
        )  # TODO Figure out deconvolution

    def forward(self, formfactor, bunch_length):
        x = self.formfactor_encoder(formfactor)
        x = torch.concatenate([x, bunch_length])
        return self.current_decoder(x)


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

        self.current_encoder = nn.Sequential(
            nn.Conv1d(), nn.Conv1d(), nn.Conv1d(), nn.Flatten()
        )
        self.mlp = nn.Sequential(nn.Linear(), nn.Linear(), nn.Linear())

    def forward(self, current, bunch_length):
        x = self.current_encoder(current)
        x = torch.concatenate([x, bunch_length])
        return self.mlp(x)
