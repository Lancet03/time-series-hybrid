"""Copied from HN-MVTS/src/models/tsmixer.py (Apache-2.0).

Original SHA256: db6e683cc89d7147337f250a305d5f1585397c4e036304719922f17320dbf371
Network code is preserved; imports are package-relative.
"""
import torch
import torch.nn as nn

from .neural import NeuralModel
from .layers import ReversibleInstanceNormalization as RIN
from .layers import MLPLayerNorm
from .layers import HyperLinearLayer


class MLPMixerLayer(nn.Module):

    def __init__(self, d_input, d_hidden_temporal, d_hidden_channel, n_input):
        super().__init__()
        self.channel_mlp = MLPLayerNorm(n_input, n_input, d_hidden_channel)
        self.temporal_mlp = MLPLayerNorm(d_input, d_input, d_hidden_temporal)

    def forward(self, X):
        X = X + self.temporal_mlp(X)
        X = X + self.channel_mlp(X.transpose(1, 2)).transpose(1, 2)
        return X


class TSMixer(nn.Module):

    def __init__(self, d_input, d_output, d_hidden_temporal, d_hidden_channel, n_input, n_blocks=2):
        super().__init__()
        self.instance_norm = RIN(n_input)
        self.mixer = torch.nn.Sequential()
        for i in range(n_blocks):
            self.mixer.add_module("mlpmixer_layer_{}".format(i), MLPMixerLayer(d_input, d_hidden_temporal, d_hidden_channel, n_input))
        self.projection = nn.Linear(d_input, d_output)

    def forward(self, X):
        X = self.instance_norm(X.transpose(1, 2), "norm").transpose(1, 2)
        X = self.mixer(X)
        X = self.projection(X)
        X = self.instance_norm(X.transpose(1, 2), "denorm").transpose(1, 2)
        return X


class TSMixerHyper(nn.Module):

    def __init__(self, d_input, d_output, d_hidden_temporal, d_hidden_channel, d_hyper_hidden, n_input, d_embedding=8, embedding=None, n_blocks=2):
        super().__init__()
        self.n_input = n_input
        self.d_output = d_output
        self.instance_norm = RIN(n_input)
        self.mixer = torch.nn.Sequential()
        for i in range(n_blocks):
            self.mixer.add_module("mlpmixer_layer_{}".format(i), MLPMixerLayer(d_input, d_hidden_temporal, d_hidden_channel, n_input))
        self.projection_hyper = HyperLinearLayer(d_input, d_output, d_hyper_hidden, n_input, d_embedding, embedding)

    def forward(self, X):
        X = self.instance_norm(X.transpose(1, 2), "norm").transpose(1, 2)
        X = self.mixer(X)
        X = self.projection_hyper(X)
        X = self.instance_norm(X.transpose(1, 2), "denorm").transpose(1, 2)
        return X


class TSMixerModel(NeuralModel):
    """Common evaluator adapter for the HN-MVTS TSMixer Base network."""

    def __init__(self, context, horizon, channels, seed=0, device="cpu", **params):
        super().__init__("tsmixer", context, horizon, channels, seed=seed, device=device, **params)
