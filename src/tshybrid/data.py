"""Chronological splits and lazy windows. An origin is the first unseen index."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_origins(start, end, context, horizon, stride=1, limit=None, legacy=False):
    if min(context, horizon, stride) < 1 or start < context:
        raise ValueError("Positive context/horizon/stride and enough history are required")
    # HN-MVTS slices XY by window START indices. Adding context converts
    # those indices to target origins, yielding the same bounds as common.
    # Keep legacy for caller compatibility; protocol differences are in scoring/training.
    stop = end - horizon + 1
    values = np.arange(start, stop, stride, dtype=np.int64)
    if not len(values):
        raise ValueError(f"No complete windows: start={start}, end={end}, H={horizon}")
    if limit is not None:
        if limit < 1:
            raise ValueError("Origin limit must be positive")
        if len(values) > limit:
            values = values[np.linspace(0, len(values) - 1, limit, dtype=int)]
    return values


class Windows(Dataset):
    def __init__(self, values, origins, context, horizon):
        self.values, self.origins = values, origins
        self.context, self.horizon = context, horizon

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        t = int(self.origins[index])
        # Only a single window is copied, never a full [windows, channels, context] array.
        return (self.values[t-self.context:t].T.copy(), self.values[t:t+self.horizon].T.copy())


@dataclass
class SeriesData:
    name: str
    raw: np.ndarray
    values: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    mase_scale: np.ndarray
    train_end: int
    val_end: int
    seasonality: int
    channels: list
    manifest: dict


def load_dataset(spec, config_dir):
    path = (Path(config_dir) / spec["path"]).resolve()
    if path.suffix == ".npy":
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        if raw.ndim != 2:
            raise ValueError(f"Expected [time, channels]: {path}")
        channels = [str(i) for i in range(raw.shape[1])]
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
        date_column = spec.get("date_column", "date")
        if date_column in frame:
            dates = pd.to_datetime(frame.pop(date_column), errors="raise")
            if not dates.is_monotonic_increasing or dates.duplicated().any():
                raise ValueError("CSV timestamps must be unique and increasing")
            if len(dates) > 2 and dates.diff().iloc[1:].nunique() != 1:
                raise ValueError("Irregular timestamps: resample explicitly before evaluation")
        raw = frame.to_numpy(dtype=np.float64)
        channels = list(frame.columns)
    else:
        raise ValueError("Supported dataset formats: .npy or numeric wide .csv")
    original_shape = list(raw.shape)
    if spec.get("take_n"):
        raw = raw[:spec["take_n"]]
    indices = spec.get("channels", list(range(raw.shape[1])))
    if len(set(indices)) != len(indices) or not indices:
        raise ValueError("Channel indices must be unique and nonempty")
    channels = [channels[i] for i in indices]
    raw = np.array(raw[:, indices], dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite observations: specify and apply a causal missing-data policy first")
    fractions = spec["split"]
    if len(fractions) != 3 or min(fractions) <= 0 or not np.isclose(sum(fractions), 1):
        raise ValueError("split must be positive train/validation/test fractions summing to one")
    train_end = int(len(raw) * fractions[0])
    test_size = int(len(raw) * fractions[2])
    val_end = len(raw) - test_size
    mean = raw[:train_end].mean(axis=0)
    scale = raw[:train_end].std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    seasonality = int(spec["seasonality"])
    if not 1 <= seasonality < train_end:
        raise ValueError("Seasonality must be positive and shorter than train")
    mase = np.abs(raw[seasonality:train_end] - raw[:train_end-seasonality]).mean(axis=0)
    mase = np.where(mase > 1e-12, mase, np.nan)
    values = ((raw - mean) / scale).astype(np.float32)
    manifest = {**spec, "resolved_path": str(path), "sha256": file_hash(path),
                "original_shape": original_shape, "used_shape": list(raw.shape), "channel_names": channels,
                "train_end": train_end, "val_end": val_end, "test_end": len(raw),
                "mean": mean.tolist(), "scale": scale.tolist(),
                "mase_scale": [float(v) if np.isfinite(v) else None for v in mase]}
    return SeriesData(spec["name"], raw, values, mean, scale, mase, train_end, val_end,
                      seasonality, channels, manifest)
