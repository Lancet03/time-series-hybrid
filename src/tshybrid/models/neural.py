"""Train the project's own neural baselines on lazy multichannel windows."""
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data import Windows, make_origins
from .base import Forecast, Model
from .layers import RevIN
from .moderntcn import ModernTCNNet
from .networks import DLinearNet, ITransformerNet, PatchTSTNet, TSMixerNet


NEURAL_MODELS = {
    "dlinear": (DLinearNet, {}, 50, 1e-4),
    "tsmixer": (TSMixerNet, {"d_hidden_temporal": 128, "d_hidden_channel": 24}, 25, 1e-4),
    "moderntcn": (ModernTCNNet, {"d_hidden": 256}, 15, 1e-4),
    "patchtst": (PatchTSTNet, {"d_hidden": 32}, 10, 2e-4),
    "itransformer": (ITransformerNet, {"d_hidden": 256}, 25, 2e-4),
}


class LinearNet(nn.Module):
    """Additional single-projection control."""

    def __init__(self, context, horizon, channels, revin=True):
        super().__init__()
        self.instance_norm = RevIN(channels) if revin else None
        self.linear_season = nn.Linear(context, horizon)

    def forward(self, x):
        if self.instance_norm is not None:
            x = self.instance_norm(x.transpose(1, 2), "norm").transpose(1, 2)
        out = self.linear_season(x)
        if self.instance_norm is not None:
            out = self.instance_norm(out.transpose(1, 2), "denorm").transpose(1, 2)
        return out


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class NeuralModel(Model):
    def __init__(self, architecture, context, horizon, channels, seed=0, device="cpu", epochs=None,
                 batch_size=64, lr=None, revin=True, kernel=25, train_stride=1,
                 train_limit=None, model_params=None):
        seed_everything(seed)
        params = dict(model_params or {})
        if architecture == "linear":
            if params:
                raise ValueError("The Linear control has no model_params")
            net = LinearNet(context, horizon, channels, revin)
            default_epochs, default_lr = 50, 1e-4
        else:
            if architecture not in NEURAL_MODELS:
                raise ValueError(f"Unknown neural architecture: {architecture}")
            if not revin:
                raise ValueError("These neural baselines require RevIN (revin=True)")
            network, defaults, default_epochs, default_lr = NEURAL_MODELS[architecture]
            params = {**defaults, **params}
            if {"d_input", "d_output", "n_input"} & params.keys():
                raise ValueError("context, horizon and channels are supplied by the experiment")
            if architecture == "dlinear":
                params.setdefault("kernel_size", kernel)
                if params["kernel_size"] < 1 or params["kernel_size"] % 2 != 1:
                    raise ValueError("Moving average kernel must be positive and odd")
            net = network(d_input=context, d_output=horizon, n_input=channels, **params)
        epochs = default_epochs if epochs is None else epochs
        lr = default_lr if lr is None else lr
        if epochs < 1 or batch_size < 1 or lr <= 0:
            raise ValueError("Positive epochs, batch_size and lr are required")
        self.net = net.to(device)
        self.architecture = architecture
        self.context, self.horizon, self.device = context, horizon, device
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.train_stride, self.train_limit = train_stride, train_limit
        self.history = []
        self.metadata = {"point_forecast": "MSE-trained output", "revin": revin,
                         "epochs": epochs, "lr": lr, "batch_size": batch_size,
                         "train_stride": train_stride, "train_limit": train_limit,
                         "linear_only": architecture == "linear", "device": device,
                         "architecture": architecture, "model_params": params,
                         "network_class": f"{type(net).__module__}.{type(net).__name__}",
                         "parameter_count": sum(p.numel() for p in net.parameters()),
                         "implementation": "time-series-hybrid"}

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
            raise ValueError("Neural forecast horizon differs from training horizon")
        self.net.eval()
        with torch.inference_mode():
            point = self.net(torch.as_tensor(x, dtype=torch.float32, device=self.device)).cpu().numpy()
        return Forecast(point)


class DLinear(NeuralModel):
    """DLinear training interface backed by the local DLinearNet."""

    def __init__(self, context, horizon, channels, seed=0, device="cpu", epochs=50,
                 batch_size=64, lr=1e-4, revin=True, kernel=25, train_stride=1,
                 train_limit=None, linear=False, model_params=None):
        super().__init__("linear" if linear else "dlinear", context, horizon, channels, seed, device,
                         epochs, batch_size, lr, revin, kernel, train_stride, train_limit, model_params)
