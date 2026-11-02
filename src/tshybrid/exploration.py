"""Dataset exploration with explicit split boundaries and train-only reference scales."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.signal import periodogram
from statsmodels.tsa.stattools import acf


def load_catalog(config_path):
    """Read configured NPY arrays without changing, imputing or standardizing them."""
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    catalog = {}
    for spec in config["datasets"]:
        path = (config_path.parent / spec["path"]).resolve()
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        if raw.ndim != 2:
            raise ValueError(f"Expected [time, channels]: {path}")
        shape = raw.shape
        used = raw[:spec.get("take_n") or len(raw)]
        indices = spec.get("channels", list(range(raw.shape[1])))
        if indices != list(range(raw.shape[1])):
            used = used[:, indices]
        fractions = spec["split"]
        if len(fractions) != 3 or min(fractions) <= 0 or not np.isclose(sum(fractions), 1):
            raise ValueError("Invalid chronological split")
        train_end = int(len(used) * fractions[0])
        val_end = len(used) - int(len(used) * fractions[2])
        catalog[spec["name"]] = dict(
            spec=spec, path=path, raw=used, original_shape=shape,
            channels=[str(i) for i in indices], train_end=train_end, val_end=val_end,
        )
    return config, catalog


def split_arrays(item, include_test=False):
    raw, train_end, val_end = item["raw"], item["train_end"], item["val_end"]
    parts = {"train": raw[:train_end], "validation": raw[train_end:val_end]}
    if include_test:
        parts["test (post-hoc)"] = raw[val_end:]
    return parts


def train_reference(item):
    train = np.asarray(item["raw"][:item["train_end"]], dtype=float)
    finite = np.where(np.isfinite(train), train, np.nan)
    mean, std = np.nanmean(finite, axis=0), np.nanstd(finite, axis=0)
    return mean, np.where(std < 1e-12, 1.0, std)


def inventory(catalog):
    rows = []
    for name, item in catalog.items():
        n, channels = item["raw"].shape
        rows.append(dict(dataset=name, file=item["path"].name,
                         original_rows=item["original_shape"][0], used_rows=n, channels=channels,
                         frequency=item["spec"].get("frequency"),
                         seasonality=item["spec"]["seasonality"], train=item["train_end"],
                         validation=item["val_end"]-item["train_end"], test=n-item["val_end"]))
    return pd.DataFrame(rows)


def channel_profile(item, include_test=False):
    """Outlier flags use train IQR fences; flags are descriptive, not data errors."""
    train = np.asarray(item["raw"][:item["train_end"]], dtype=float)
    clean_train = np.where(np.isfinite(train), train, np.nan)
    q1, q3 = np.nanquantile(clean_train, [.25, .75], axis=0)
    lo, hi = q1 - 3*(q3-q1), q3 + 3*(q3-q1)
    rows = []
    for split, values in split_arrays(item, include_test).items():
        for j, channel in enumerate(item["channels"]):
            x = np.asarray(values[:, j], dtype=float)
            valid = np.isfinite(x)
            v = x[valid]
            pairs = valid[1:] & valid[:-1]
            rows.append(dict(
                split=split, channel=channel, n=len(x), nonfinite_pct=100*(~valid).mean(),
                mean=v.mean() if len(v) else np.nan, std=v.std() if len(v) else np.nan,
                min=v.min() if len(v) else np.nan, median=np.median(v) if len(v) else np.nan,
                max=v.max() if len(v) else np.nan,
                zero_pct=100*np.mean(v == 0) if len(v) else np.nan,
                repeat_pct=100*np.mean((x[1:] == x[:-1])[pairs]) if pairs.any() else np.nan,
                outside_train_fence_pct=100*np.mean((v < lo[j]) | (v > hi[j])) if len(v) else np.nan,
                constant=bool(len(v) and np.ptp(v) < 1e-12),
            ))
    return pd.DataFrame(rows)


def split_drift(item, include_test=False):
    """Measure shifts against train; never fit normalization on validation or test."""
    mean, scale = train_reference(item)
    profile = channel_profile(item, include_test)
    lookup = {c: j for j, c in enumerate(item["channels"])}
    profile["mean_shift_train_std"] = [
        (r.mean - mean[lookup[r.channel]]) / scale[lookup[r.channel]]
        for r in profile.itertuples()
    ]
    # Ratios to truly constant train channels are undefined.
    train_std = profile[profile.split == "train"].set_index("channel")["std"]
    profile["std_ratio_to_train"] = [
        r.std / train_std[r.channel] if train_std[r.channel] > 1e-12 else np.nan
        for r in profile.itertuples()
    ]
    return profile[["split", "channel", "mean_shift_train_std", "std_ratio_to_train",
                    "outside_train_fence_pct"]]


def autocorrelations(values, channels, lags):
    """Keep time spacing: gaps yield NaN rather than silently removing observations."""
    rows = []
    lags = sorted(set(int(lag) for lag in lags if 0 < lag < len(values)))
    for j, channel in enumerate(channels):
        x = np.asarray(values[:, j], dtype=float)
        valid = np.isfinite(x).all() and np.std(x) > 1e-12
        coefficients = acf(x, nlags=max(lags), fft=True) if valid and lags else None
        for lag in lags:
            rows.append(dict(channel=channel, lag=lag,
                             acf=float(coefficients[lag]) if coefficients is not None else np.nan))
    return pd.DataFrame(rows, columns=["channel", "lag", "acf"])


def spectral_peaks(values, channels, top_n=3, max_period=None):
    """Train spectra with linear detrending. Periods are in observations, not dates."""
    rows = []
    for j, channel in enumerate(channels):
        x = np.asarray(values[:, j], dtype=float)
        if not np.isfinite(x).all() or np.std(x) < 1e-12:
            continue
        frequencies, power = periodogram(x, detrend="linear")
        positive = frequencies > 0
        periods, power = 1/frequencies[positive], power[positive]
        mask = (periods >= 2) & (periods <= (max_period or len(x)/4))
        indices = np.flatnonzero(mask)
        best = indices[np.argsort(power[indices])[-top_n:][::-1]]
        for rank, k in enumerate(best, 1):
            rows.append(dict(channel=channel, rank=rank, period_steps=periods[k],
                             power_share=power[k]/power.sum() if power.sum() else np.nan))
    return pd.DataFrame(rows)


def seasonal_profile(values, period):
    frame = pd.DataFrame(values)
    return frame.groupby(np.arange(len(frame)) % period).mean()


def naive_validation(item, horizons=(48, 96), stride=None):
    """Rolling-origin validation: predictions use only observations before each origin."""
    raw = np.asarray(item["raw"], dtype=float)
    _, scale = train_reference(item)
    period = int(item["spec"]["seasonality"])
    rows = []
    for horizon in horizons:
        origins = np.arange(max(item["train_end"], period), item["val_end"]-horizon+1,
                            stride or horizon)
        if not len(origins):
            continue
        lead = np.arange(horizon)
        truth = raw[origins[:, None] + lead]
        predictions = {
            "persistence": np.broadcast_to(raw[origins-1, None, :], truth.shape),
            "seasonal_naive": raw[origins[:, None]-period + lead % period],
        }
        for model, prediction in predictions.items():
            error = (prediction-truth)/scale
            if not np.isfinite(error).all():
                raise ValueError("Naive validation requires finite data; define a missing-data policy")
            for j, channel in enumerate(item["channels"]):
                rows.append(dict(horizon=horizon, model=model, channel=channel, origins=len(origins),
                                 mse=np.mean(error[..., j]**2), mae=np.mean(np.abs(error[..., j]))))
    return pd.DataFrame(rows)


def arima_train_diagnostics(item, context=336, warmup=128, horizon=96,
                            n_origins=16, order=(2, 0, 0)):
    """Match the experiment's ARIMA prefix/filtering on windows wholly inside train.

    Future train targets are diagnostic holdouts, never inputs to the ARIMA fit.
    Residual ACF is computed within each window, never across window boundaries.
    """
    from .models.arima import _fit_one

    if not 0 < warmup < context or item["train_end"] < context+horizon:
        raise ValueError("Insufficient train history or invalid warmup/context")
    mean, scale = train_reference(item)
    train = (np.asarray(item["raw"][:item["train_end"]], dtype=float)-mean)/scale
    if not np.isfinite(train).all():
        raise ValueError("ARIMA diagnostics require finite train data")
    origins = np.unique(np.linspace(context, len(train)-horizon, n_origins, dtype=int))
    diagnostics, predictions, residual_acf, example = [], [], [], []
    for origin in origins:
        for j, channel in enumerate(item["channels"]):
            history = train[origin-context:origin, j]
            base, residuals, nonconverged, _ = _fit_one(
                history, horizon, order, warmup, 100, "default")
            coefficients = acf(residuals, nlags=min(len(residuals)-1, 96), fft=True) \
                if np.std(residuals) > 1e-12 else None
            for lag in (1, 24, 48, 96):
                if lag < len(residuals):
                    residual_acf.append(dict(origin=origin, channel=channel, lag=lag,
                                             acf=coefficients[lag] if coefficients is not None else np.nan))
            diagnostics.append(dict(origin=origin, channel=channel, nonconverged=nonconverged,
                                    residual_mean=residuals.mean(), residual_std=residuals.std(),
                                    history_std=history.std()))
            truth = train[origin:origin+horizon, j]
            for model, correction in {"ARIMA": np.zeros(horizon),
                                      "ARIMA + last residual": np.full(horizon, residuals[-1]),
                                      "ARIMA + mean residual": np.full(horizon, residuals.mean())}.items():
                error = base+correction-truth
                predictions.append(dict(origin=origin, channel=channel, model=model,
                                        mse=np.mean(error**2), mae=np.mean(np.abs(error))))
            if origin == origins[0]:
                for lead in range(horizon):
                    example.append(dict(channel=channel, lead=lead+1, truth=truth[lead], base=base[lead],
                                        required_correction=truth[lead]-base[lead],
                                        last_residual=residuals[-1], mean_residual=residuals.mean()))
    return {"fits": pd.DataFrame(diagnostics), "scores": pd.DataFrame(predictions),
            "acf": pd.DataFrame(residual_acf), "example": pd.DataFrame(example)}
