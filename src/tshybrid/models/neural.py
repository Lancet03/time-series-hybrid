"""DLinear with optional affine RevIN, trained on lazy multichannel windows.

The decomposition/RevIN equations follow HN-MVTS (Apache-2.0); see THIRD_PARTY.md.
This is a separate implementation; source parity is checked by a numerical test.
"""
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data import Windows, make_origins
from .base import Forecast, Model


class DLinearNet(nn.Module):
    def __init__(self, context, horizon, channels, revin=True, kernel=25, linear=False):
        super().__init__()
        self.revin, self.kernel, self.linear = revin, kernel, linear
        if kernel % 2 != 1:
            raise ValueError("Moving average kernel must be odd")
        if revin:
            self.affine_weight = nn.Parameter(torch.ones(channels))
            self.affine_bias = nn.Parameter(torch.zeros(channels))
        self.linear_season = nn.Linear(context, horizon)
        if not linear:
            self.linear_trend = nn.Linear(context, horizon)

    def forward(self, x):
        if self.revin:
            mean = x.mean(-1, keepdim=True).detach()
            std = torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-5).detach()
            x = (x - mean) / std
            x = x * self.affine_weight[None, :, None] + self.affine_bias[None, :, None]
        if self.linear:
            out = self.linear_season(x)
        else:
            padded = nn.functional.pad(x, ((self.kernel-1)//2, (self.kernel-1)//2), mode="replicate")
            trend = nn.functional.avg_pool1d(padded, self.kernel, stride=1)
            out = self.linear_season(x - trend) + self.linear_trend(trend)
        if self.revin:
            out = (out - self.affine_bias[None, :, None]) / (self.affine_weight[None, :, None] + 1e-10)
            out = out * std + mean
        return out


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DLinear(Model):
    def __init__(self, context, horizon, channels, seed=0, device="cpu", epochs=50,
                 batch_size=64, lr=1e-4, revin=True, kernel=25, train_stride=1,
                 train_limit=None, linear=False):
        seed_everything(seed)
        self.net = DLinearNet(context, horizon, channels, revin, kernel, linear).to(device)
        self.context, self.horizon, self.device = context, horizon, device
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.train_stride, self.train_limit = train_stride, train_limit
        self.history = []
        self.metadata = {"point_forecast": "MSE-trained output", "revin": revin,
                         "epochs": epochs, "lr": lr, "batch_size": batch_size,
                         "train_stride": train_stride, "train_limit": train_limit,
                         "linear_only": linear, "device": device}

    def fit(self, data, val_origins, protocol):
        train_origins = make_origins(self.context, data.train_end, self.context, self.horizon,
                                    self.train_stride, self.train_limit)
        train = DataLoader(Windows(data.values, train_origins, self.context, self.horizon),
                           batch_size=self.batch_size, shuffle=True, num_workers=0)
        val = DataLoader(Windows(data.values, val_origins, self.context, self.horizon),
                         batch_size=self.batch_size, shuffle=protocol == "hn_mvts_code", num_workers=0)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        best_score, best_state = float("inf"), None
        for epoch in range(self.epochs):
            self.net.train()
            total, count = 0.0, 0
            for x, y in train:
                x, y = x.to(self.device), y.to(self.device)
                loss = nn.functional.mse_loss(self.net(x), y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total += loss.item() * y.numel()
                count += y.numel()
            self.net.eval()
            error_sum, val_count = 0.0, 0
            with torch.inference_mode():
                for x, y in val:
                    x, y = x.to(self.device), y.to(self.device)
                    loss = nn.functional.mse_loss(self.net(x), y).item()
                    weight = 1 if protocol == "hn_mvts_code" else y.numel()
                    error_sum += loss * weight
                    val_count += weight
            score = error_sum / val_count
            self.history.append({"epoch": epoch, "mse_train": total/count, "mse_val": score})
            if score < best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                self.metadata["best_epoch"] = epoch
            print(f"    epoch {epoch+1}/{self.epochs}: train={total/count:.6f} val={score:.6f}", flush=True)
            if protocol == "hn_mvts_code":
                # The original evaluates test each epoch. Creating an iterator consumes
                # a torch RNG base_seed even without shuffling. Preserve RNG progression
                # without inspecting test observations during model selection.
                torch.empty((), dtype=torch.int64).random_()
        if best_state is None:
            raise ValueError("No finite validation scores or zero training epochs")
        self.net.load_state_dict(best_state)
        self.metadata["training_windows"] = len(train_origins)
        return self

    def predict(self, x, horizon):
        if horizon != self.horizon:
            raise ValueError("DLinear horizon differs from training horizon")
        self.net.eval()
        with torch.inference_mode():
            point = self.net(torch.as_tensor(x, dtype=torch.float32, device=self.device)).cpu().numpy()
        return Forecast(point)
