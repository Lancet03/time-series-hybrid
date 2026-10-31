"""Frozen, channel-independent official FM APIs; immutable HF revisions are recorded."""
import json
from pathlib import Path

import numpy as np
import torch

from .base import Forecast, Model


def resolve_revision(checkpoint, revision, cache_dir):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    options = dict(repo_id=checkpoint, revision=revision, cache_dir=cache_dir,
                   allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model"])
    try:
        path = snapshot_download(**options, local_files_only=True)
        snapshot = Path(path)
        indexes = list(snapshot.glob("*.index.json"))
        if indexes:
            weights = [snapshot / name for index in indexes
                       for name in set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())]
        else:
            weights = list(snapshot.glob("*.safetensors")) + list(snapshot.glob("*.bin"))
        # HF can return a snapshot directory containing only config.json after an
        # interrupted download. Resume the weights instead of treating it as complete.
        if not (snapshot / "config.json").is_file() or not weights or not all(p.is_file() for p in weights):
            raise LocalEntryNotFoundError("Cached checkpoint is incomplete")
    except LocalEntryNotFoundError:
        path = snapshot_download(**options)
    # snapshot_download resolves even a moving branch to a concrete commit directory.
    return path, Path(path).name


class FrozenFoundation(Model):
    """Shared checkpoint loading and batched [window, channel, lead] reshaping."""

    family = None
    package = None

    def __init__(self, checkpoint, revision, device="cpu", cache_dir=None, batch_size=16):
        from importlib.util import find_spec
        if find_spec(self.package) is None:
            raise ImportError(f"Install the {self.family!r} project extra in this environment first")
        if batch_size < 1:
            raise ValueError("Foundation batch_size must be positive")
        self.device, self.batch_size = device, batch_size
        self.path, commit = resolve_revision(checkpoint, revision, cache_dir)
        self.metadata = {"family": self.family, "checkpoint": checkpoint, "requested_revision": revision,
                         "resolved_revision": commit, "device": device, "frozen": True,
                         "channel_mode": "independent", "quantiles": [0.1, 0.5, 0.9]}
        self.load()

    def load(self):
        raise NotImplementedError

    def predict_batch(self, batch, horizon):
        raise NotImplementedError

    def predict(self, x, horizon):
        shape = (*x.shape[:-1], horizon)
        flat = x.reshape(-1, x.shape[-1])
        points, quantile_batches = [], []
        for start in range(0, len(flat), self.batch_size):
            with torch.inference_mode():
                point, quantiles = self.predict_batch(flat[start:start+self.batch_size], horizon)
            points.append(np.asarray(point)[..., :horizon])
            quantile_batches.append(np.asarray(quantiles)[:, :horizon])
        point = np.concatenate(points).reshape(shape)
        quantiles = np.concatenate(quantile_batches).reshape(*shape, 3)
        return Forecast(point, {q: quantiles[..., i] for i, q in enumerate([0.1, 0.5, 0.9])})
