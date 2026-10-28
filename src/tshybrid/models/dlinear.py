"""Copied from HN-MVTS/src/models/dlinear.py (Apache-2.0).

Original SHA256: ed0b6757672790098f18c71560b126b575025d00bb76ef4db5388ce5da8f5c89
Network code is preserved; imports are package-relative.
"""
import torch
import torch.nn as nn

from .neural import NeuralModel
from .layers import ReversibleInstanceNormalization as RIN
from .layers import SeasonalTrendDecompositionLayer as STL
from .layers import HyperLinearLayer


class DLinear(nn.Module):

    def __init__(self, d_input, d_output, n_input, kernel_size=25):
        super().__init__()
        self.instance_norm = RIN(n_input)
        self.stl = STL(kernel_size)
        self.linear_season = nn.Linear(d_input, d_output)
        self.linear_trend = nn.Linear(d_input, d_output)

    def forward(self, X):
        X = self.instance_norm(X.transpose(1, 2), "norm").transpose(1, 2)
        X_season, X_trend = self.stl(X)
        X = self.linear_season(X_season) + self.linear_trend(X_trend)
        X = self.instance_norm(X.transpose(1, 2), "denorm").transpose(1, 2)
        return X

class DLinearHyper(nn.Module):

    def __init__(self, d_input, d_output, d_hyper_hidden, n_input, d_embedding=8, embedding=None, kernel_size=25):
        super().__init__()
        self.d_output = d_output
        self.n_input = n_input
        self.instance_norm = RIN(n_input)
        self.stl = STL(kernel_size)
        self.hyperlinear = HyperLinearLayer(d_input, d_output, d_hyper_hidden, n_input*2, d_embedding, embedding)

    def forward(self, X):
        X = self.instance_norm(X.transpose(1, 2), "norm").transpose(1, 2)
        X_season, X_trend = self.stl(X)
        XX = torch.cat([X_season, X_trend], dim=1)
        ZZ = self.hyperlinear(XX)
        Z = ZZ[:,:self.n_input] + ZZ[:,self.n_input:]
        X = self.instance_norm(Z.transpose(1, 2), "denorm").transpose(1, 2)
        return X


class DLinearModel(NeuralModel):
    """Common evaluator adapter for the HN-MVTS DLinear Base network."""

    def __init__(self, context, horizon, channels, seed=0, device="cpu", **params):
        super().__init__("dlinear", context, horizon, channels, seed=seed, device=device, **params)
