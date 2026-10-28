"""Local ModernTCN baseline matching the reference topology; see THIRD_PARTY.md."""

import torch
from torch import nn

from .layers import RevIN


def conv_bn(channels, kernel_size):
    layer = nn.Sequential()
    layer.add_module("conv", nn.Conv1d(channels, channels, kernel_size,
                                     padding=kernel_size // 2, groups=channels, bias=False))
    layer.add_module("bn", nn.BatchNorm1d(channels))
    return layer


class LargeKernelConv(nn.Module):
    def __init__(self, channels, large_size=31, small_size=5):
        super().__init__()
        self.lkb_origin = conv_bn(channels, large_size)
        self.small_conv = conv_bn(channels, small_size)

    def forward(self, x):
        return self.lkb_origin(x) + self.small_conv(x)


class TemporalBlock(nn.Module):
    def __init__(self, channels, width=32, expansion=4, dropout=0.5):
        super().__init__()
        hidden = width * expansion
        self.dw = LargeKernelConv(channels * width)
        self.norm = nn.BatchNorm1d(width)
        self.ffn1pw1 = nn.Conv1d(channels * width, channels * hidden, 1, groups=channels)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(channels * hidden, channels * width, 1, groups=channels)
        self.ffn1drop1 = nn.Dropout(dropout)
        self.ffn1drop2 = nn.Dropout(dropout)
        self.ffn2pw1 = nn.Conv1d(channels * width, channels * hidden, 1, groups=width)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(channels * hidden, channels * width, 1, groups=width)
        self.ffn2drop1 = nn.Dropout(dropout)
        self.ffn2drop2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        batch, channels, width, length = x.shape
        x = self.dw(x.reshape(batch, channels * width, length))
        x = self.norm(x.reshape(batch * channels, width, length))
        x = x.reshape(batch, channels * width, length)
        x = self.ffn1act(self.ffn1drop1(self.ffn1pw1(x)))
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(batch, channels, width, length).permute(0, 2, 1, 3)
        x = x.reshape(batch, width * channels, length)
        x = self.ffn2act(self.ffn2drop1(self.ffn2pw1(x)))
        x = self.ffn2drop2(self.ffn2pw2(x))
        return residual + x.reshape(batch, width, channels, length).permute(0, 2, 1, 3)


class TemporalStage(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.blocks = nn.ModuleList([TemporalBlock(channels)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ModernTCNNet(nn.Module):
    """The reference Base uses two stages of width 32.

    d_hidden and n_blocks are retained for config compatibility: the reference
    constructor also accepts them without changing its fixed Base topology.
    Unused reference projection branches stay registered so its state_dicts and
    seeded initialization remain compatible.
    """

    def __init__(self, d_input, d_output, d_hidden, n_input, n_blocks=4):
        super().__init__()
        self.patch_size, self.patch_stride = 8, 4
        self.downsample_ratio = 2
        self.num_stage = 2
        self.instance_norm = RevIN(n_input)
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(nn.Conv1d(1, 32, self.patch_size, stride=self.patch_stride), nn.BatchNorm1d(32))
        ])
        for _ in range(3):
            self.downsample_layers.append(nn.Sequential(nn.BatchNorm1d(32), nn.Conv1d(32, 32, 2, stride=2)))
        self.stages = nn.ModuleList([TemporalStage(n_input) for _ in range(self.num_stage)])
        self.lat_layer = nn.ModuleList()
        self.smooth_layer = nn.ModuleList()
        self.up_sample_conv = nn.ModuleList()
        for _ in range(self.num_stage):
            self.lat_layer.append(nn.Conv1d(32, 32, 1))
            self.smooth_layer.append(nn.Conv1d(32, 32, 3, padding=1))
            self.up_sample_conv.append(nn.Sequential(nn.ConvTranspose1d(32, 32, 2, stride=2), nn.BatchNorm1d(32)))
        patches = d_input // self.patch_stride
        self.head_nf = 32 * ((patches + 1) // 2)
        self.projection = nn.Sequential(nn.Flatten(start_dim=-2), nn.Linear(self.head_nf, d_output), nn.Dropout(0.0))

    def model(self, x):
        x = x.unsqueeze(-2)
        for index in range(self.num_stage):
            batch, channels, width, length = x.shape
            x = x.reshape(batch * channels, width, length)
            if index == 0:
                padding = x[..., -1:].repeat(1, 1, self.patch_size - self.patch_stride)
                x = torch.cat([x, padding], dim=-1)
            elif length % self.downsample_ratio:
                missing = self.downsample_ratio - length % self.downsample_ratio
                x = torch.cat([x, x[..., -missing:]], dim=-1)
            x = self.downsample_layers[index](x)
            x = self.stages[index](x.reshape(batch, channels, x.shape[-2], x.shape[-1]))
        return x

    def forward(self, x):
        x = self.instance_norm(x.transpose(1, 2), "norm").transpose(1, 2)
        out = self.projection(self.model(x))
        return self.instance_norm(out.transpose(1, 2), "denorm").transpose(1, 2)
