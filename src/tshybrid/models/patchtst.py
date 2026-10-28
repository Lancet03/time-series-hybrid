"""Copied from HN-MVTS/src/models/patchtst.py (Apache-2.0).

Original SHA256: 1f56782ba00ecadeb1b930c456f8bc077128e43c01624e5d416ad6cfb4e300d5
Network code is preserved; imports are package-relative.
"""
import torch
import torch.nn as nn

from .neural import NeuralModel
from .layers import ReversibleInstanceNormalization as RIN
from .layers import PatchingLayer
from .layers import HyperLinearLayer


class PatchTST(nn.Module):

    def __init__(self, d_input, d_output, d_hidden, n_input, d_patch=16, stride=8, n_heads=4, n_blocks=2):
        super().__init__()
        self.n_input = n_input
        n_patches = int((d_input - d_patch) / stride + 1) + 1
        self.instance_norm = RIN(n_input)
        self.patching = PatchingLayer(d_patch, stride)
        self.embedding = nn.Linear(d_patch, d_hidden) # d_patch > d_model
        tx_enc_block = nn.TransformerEncoderLayer(d_hidden, nhead=n_heads, dim_feedforward=d_hidden*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(tx_enc_block, num_layers=n_blocks)
        self.flatten = nn.Flatten(start_dim=-2)
        self.projection = nn.Linear(d_hidden * n_patches, d_output) # d_model * n_patches > d_output

    def forward(self, X):
        X = self.instance_norm(X.transpose(1, 2), "norm").transpose(1, 2)
        X = self.patching(X)
        X = self.embedding(X)
        X = torch.reshape(X, [X.shape[0]*X.shape[1], X.shape[-2], X.shape[-1]]) # (-1, X.shape[-2], X.shape[-1])
        X = self.transformer(X)
        X = torch.reshape(X, [-1, self.n_input, X.shape[-2], X.shape[-1]])
        X = self.flatten(X)
        X = self.projection(X)
        X = self.instance_norm(X.transpose(1, 2), "denorm").transpose(1, 2)
        return X


class PatchTSTHyper(nn.Module):

    def __init__(self, d_input, d_output, d_hidden, d_hyper_hidden, n_input, d_patch=16, stride=8, d_embedding=8, embedding=None, n_heads=4, n_blocks=2):
        super().__init__()
        self.n_input = n_input
        n_patches = int((d_input - d_patch) / stride + 1) + 1
        self.instance_norm = RIN(n_input)
        self.patching = PatchingLayer(d_patch, stride)
        self.embedding = nn.Linear(d_patch, d_hidden) # d_patch > d_model
        tx_enc_block = nn.TransformerEncoderLayer(d_hidden, nhead=n_heads, dim_feedforward=d_hidden*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(tx_enc_block, num_layers=n_blocks)
        self.flatten = nn.Flatten(start_dim=-2)
        self.projection_hyper = HyperLinearLayer(d_hidden * n_patches, d_output, d_hyper_hidden, n_input, d_embedding, embedding)

    def forward(self, X):
        X = self.instance_norm(X.transpose(1, 2), "norm").transpose(1, 2)
        X = self.patching(X)
        X = self.embedding(X)
        X = torch.reshape(X, [X.shape[0]*X.shape[1], X.shape[-2], X.shape[-1]]) # (-1, X.shape[-2], X.shape[-1])
        X = self.transformer(X)
        X = torch.reshape(X, [-1, self.n_input, X.shape[-2], X.shape[-1]])
        X = self.flatten(X)
        X = self.projection_hyper(X)
        X = self.instance_norm(X.transpose(1, 2), "denorm").transpose(1, 2)
        return X


class PatchTSTModel(NeuralModel):
    """Common evaluator adapter for the HN-MVTS PatchTST Base network."""

    def __init__(self, context, horizon, channels, seed=0, device="cpu", **params):
        super().__init__("patchtst", context, horizon, channels, seed=seed, device=device, **params)
