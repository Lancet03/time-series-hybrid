"""Shared trainer for the copied HN-MVTS networks on lazy multichannel windows."""
import random
import csv
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data import Windows, make_origins
from .base import Forecast, Model
from ..progress import Progress, duration


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
        from .registry import NEURAL_MODELS
        if architecture not in NEURAL_MODELS:
            raise ValueError(f"Unknown neural architecture: {architecture}")
        if not revin:
            raise ValueError("HN-MVTS networks require RevIN (revin=True)")
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
        self.pin_memory = torch.device(device).type == "cuda"
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.train_stride, self.train_limit = train_stride, train_limit
        self.history = []
        self.progress_callback = None
        self.history_path = None
        self.metadata = {"point_forecast": "MSE-trained output", "revin": revin,
                         "epochs": epochs, "lr": lr, "batch_size": batch_size,
                         "train_stride": train_stride, "train_limit": train_limit,
                         "device": device,
                         "pin_memory": self.pin_memory, "non_blocking_transfer": self.pin_memory,
                         "architecture": architecture, "model_params": params,
                         "network_class": f"{type(net).__module__}.{type(net).__name__}",
                         "parameter_count": sum(p.numel() for p in net.parameters()),
                         "implementation": "HN-MVTS",
                         "source": f"HN-MVTS/src/models/{architecture}.py",
                         "variant": "Base"}

    def fit(self, data, val_origins, protocol):
        train_origins = make_origins(self.context, data.train_end, self.context, self.horizon,
                                    self.train_stride, self.train_limit)
        train = DataLoader(Windows(data.values, train_origins, self.context, self.horizon),
                           batch_size=self.batch_size, shuffle=True, num_workers=0,
                           pin_memory=self.pin_memory)
        val = DataLoader(Windows(data.values, val_origins, self.context, self.horizon),
                         batch_size=self.batch_size, shuffle=protocol == "hn_mvts_code", num_workers=0,
                         pin_memory=self.pin_memory)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        best_score, best_state = float("inf"), None
        fit_started = perf_counter()
        for epoch in range(self.epochs):
            epoch_started = perf_counter()
            context = {"epoch": epoch + 1, "epochs": self.epochs}
            progress = Progress(len(train), f"epoch {epoch+1}/{self.epochs} train batches",
                                self.progress_callback, **context)
            self.net.train()
            total, count = torch.zeros((), dtype=torch.float64, device=self.device), 0
            for batch_number, (x, y) in enumerate(train, 1):
                x = x.to(self.device, non_blocking=self.pin_memory)
                y = y.to(self.device, non_blocking=self.pin_memory)
                loss = nn.functional.mse_loss(self.net(x), y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # Keep logging accumulators on device; .item() on every batch
                # synchronizes CUDA and prevents CPU/GPU overlap.
                total += loss.detach().to(torch.float64) * y.numel()
                count += y.numel()
                progress.update(batch_number)
            self.net.eval()
            error_sum = torch.zeros((), dtype=torch.float64, device=self.device)
            val_count = 0
            progress = Progress(len(val), f"epoch {epoch+1}/{self.epochs} validation batches",
                                self.progress_callback, **context)
            with torch.inference_mode():
                for batch_number, (x, y) in enumerate(val, 1):
                    x = x.to(self.device, non_blocking=self.pin_memory)
                    y = y.to(self.device, non_blocking=self.pin_memory)
                    loss = nn.functional.mse_loss(self.net(x), y)
                    weight = 1 if protocol == "hn_mvts_code" else y.numel()
                    error_sum += loss.to(torch.float64) * weight
                    val_count += weight
                    progress.update(batch_number)
            total = total.item()
            score = error_sum.item() / val_count
            epoch_seconds = perf_counter() - epoch_started
            elapsed = perf_counter() - fit_started
            eta = elapsed / (epoch + 1) * (self.epochs - epoch - 1)
            row = {"epoch": epoch, "mse_train": total/count, "mse_val": score,
                   "epoch_seconds": epoch_seconds, "elapsed_seconds": elapsed, "eta_seconds": eta}
            self.history.append(row)
            if self.history_path is not None:
                with self.history_path.open("w" if epoch == 0 else "a", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    if epoch == 0:
                        writer.writeheader()
                    writer.writerow(row)
            if score < best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                self.metadata["best_epoch"] = epoch
            print(f"    epoch {epoch+1}/{self.epochs}: train={total/count:.6f} val={score:.6f}"
                  f" | epoch time={duration(epoch_seconds)} | fit ETA={duration(eta)}"
                  f" | best epoch={self.metadata.get('best_epoch', 0)+1}", flush=True)
            if self.progress_callback:
                self.progress_callback({**context, "stage": "epoch completed", "completed": epoch+1,
                                        "total": self.epochs, "elapsed_seconds": elapsed,
                                        "eta_seconds": eta, "epoch_seconds": epoch_seconds,
                                        "mse_train": total/count, "mse_val": score,
                                        "best_epoch": self.metadata.get("best_epoch", 0)+1})
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
