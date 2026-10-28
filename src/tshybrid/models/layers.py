"""Shared layers for the project's neural baselines; attribution: THIRD_PARTY.md."""

import torch
from torch import nn
from torch.nn import functional as F


class RevIN(nn.Module):
    """Normalize each history independently and invert using its saved statistics."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(channels))
        self.affine_bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x, mode):
        # Input layout is [batch, time, channels].
        if mode == "norm":
            axes = tuple(range(1, x.ndim - 1))
            self.mean = x.mean(dim=axes, keepdim=True).detach()
            self.std = (x.var(dim=axes, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            return (x - self.mean) / self.std * self.affine_weight + self.affine_bias
        if mode == "denorm":
            return (x - self.affine_bias) / (self.affine_weight + self.eps ** 2) * self.std + self.mean
        raise ValueError(f"Unknown RevIN mode: {mode}")


class MovingAverage(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("Moving average kernel must be positive and odd")
        self.padding = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size, stride=1)

    def forward(self, x):
        return self.avg(F.pad(x, (self.padding, self.padding), mode="replicate"))


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_average = MovingAverage(kernel_size)

    def forward(self, x):
        trend = self.moving_average(x)
        return x - trend, trend


class LinearLayer(nn.Module):
    """Keep the projection names compatible with saved TSMixer checkpoints."""

    def __init__(self, d_input, d_output):
        super().__init__()
        self.linear = nn.Linear(d_input, d_output)

    def forward(self, x):
        return self.linear(x)


class MixerMLP(nn.Module):
    def __init__(self, d_input, d_output, d_hidden):
        super().__init__()
        self.ln = nn.LayerNorm(d_input)
        self.linear_in = LinearLayer(d_input, d_hidden)
        self.linear_out = LinearLayer(d_hidden, d_output)

    def forward(self, x):
        return self.linear_out(F.relu(self.linear_in(self.ln(x))))


class Patching(nn.Module):
    def __init__(self, size, stride):
        super().__init__()
        self.size, self.stride = size, stride

    def forward(self, x):
        x = F.pad(x, (0, self.stride), mode="replicate")
        return x.unfold(-1, self.size, self.stride)
