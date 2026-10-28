"""Local DLinear, TSMixer, PatchTST and iTransformer implementations.

The baseline architectures match the HN-MVTS reference setup, including RevIN.
No modules are imported from that repository. See THIRD_PARTY.md for attribution.
All networks map [batch, channels, context] to [batch, channels, horizon].
"""

from torch import nn

from .layers import MixerMLP, Patching, RevIN, SeriesDecomposition


class DLinearNet(nn.Module):
    def __init__(self, d_input, d_output, n_input, kernel_size=25):
        super().__init__()
        self.instance_norm = RevIN(n_input)
        self.stl = SeriesDecomposition(kernel_size)
        self.linear_season = nn.Linear(d_input, d_output)
        self.linear_trend = nn.Linear(d_input, d_output)

    def forward(self, x):
        x = self.instance_norm(x.transpose(1, 2), "norm").transpose(1, 2)
        season, trend = self.stl(x)
        out = self.linear_season(season) + self.linear_trend(trend)
        return self.instance_norm(out.transpose(1, 2), "denorm").transpose(1, 2)


class MixerBlock(nn.Module):
    def __init__(self, d_input, d_hidden_temporal, d_hidden_channel, n_input):
        super().__init__()
        self.channel_mlp = MixerMLP(n_input, n_input, d_hidden_channel)
        self.temporal_mlp = MixerMLP(d_input, d_input, d_hidden_temporal)

    def forward(self, x):
        x = x + self.temporal_mlp(x)
        return x + self.channel_mlp(x.transpose(1, 2)).transpose(1, 2)


class TSMixerNet(nn.Module):
    def __init__(self, d_input, d_output, d_hidden_temporal, d_hidden_channel, n_input, n_blocks=2):
        super().__init__()
        self.instance_norm = RevIN(n_input)
        self.mixer = nn.Sequential()
        for index in range(n_blocks):
            self.mixer.add_module(f"mlpmixer_layer_{index}",
                                  MixerBlock(d_input, d_hidden_temporal, d_hidden_channel, n_input))
        self.projection = nn.Linear(d_input, d_output)

    def forward(self, x):
        x = self.instance_norm(x.transpose(1, 2), "norm").transpose(1, 2)
        out = self.projection(self.mixer(x))
        return self.instance_norm(out.transpose(1, 2), "denorm").transpose(1, 2)


class PatchTSTNet(nn.Module):
    def __init__(self, d_input, d_output, d_hidden, n_input, d_patch=16, stride=8, n_heads=4, n_blocks=2):
        super().__init__()
        self.n_input = n_input
        patches = (d_input - d_patch) // stride + 2
        self.instance_norm = RevIN(n_input)
        self.patching = Patching(d_patch, stride)
        self.embedding = nn.Linear(d_patch, d_hidden)
        encoder = nn.TransformerEncoderLayer(d_hidden, n_heads, dim_feedforward=d_hidden * 4,
                                             batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, n_blocks)
        self.flatten = nn.Flatten(start_dim=-2)
        self.projection = nn.Linear(d_hidden * patches, d_output)

    def forward(self, x):
        x = self.instance_norm(x.transpose(1, 2), "norm").transpose(1, 2)
        x = self.embedding(self.patching(x))
        batch, channels, patches, width = x.shape
        x = self.transformer(x.reshape(batch * channels, patches, width))
        out = self.projection(self.flatten(x.reshape(batch, channels, patches, width)))
        return self.instance_norm(out.transpose(1, 2), "denorm").transpose(1, 2)


class ITransformerNet(nn.Module):
    def __init__(self, d_input, d_output, d_hidden, n_input, n_heads=4, n_blocks=2):
        super().__init__()
        self.instance_norm = RevIN(n_input)
        self.embedding = nn.Linear(d_input, d_hidden)
        encoder = nn.TransformerEncoderLayer(d_hidden, n_heads, dim_feedforward=d_hidden * 2,
                                             batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, n_blocks)
        self.projection = nn.Linear(d_hidden, d_output)

    def forward(self, x):
        x = self.instance_norm(x.transpose(1, 2), "norm").transpose(1, 2)
        out = self.projection(self.transformer(self.embedding(x)))
        return self.instance_norm(out.transpose(1, 2), "denorm").transpose(1, 2)
