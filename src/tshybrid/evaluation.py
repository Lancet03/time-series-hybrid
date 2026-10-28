"""Streaming metrics and predictions with element-weighted aggregation."""
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .data import Windows
from .progress import Progress


class Scores:
    def __init__(self, horizon):
        self.horizon = horizon
        self.sums, self.counts = {}, {}
        self.batch_mse, self.batch_mae = [], []

    def _add(self, name, values):
        finite = np.isfinite(values)
        self.sums[name] = self.sums.get(name, 0) + np.where(finite, values, 0).sum(axis=(0, 1))
        self.counts[name] = self.counts.get(name, 0) + finite.sum(axis=(0, 1))

    def update(self, truth, forecast, scale, mase_scale):
        error = forecast.point.astype(np.float64) - truth.astype(np.float64)
        raw_error = error * scale[None, :, None]
        self._add("mse", error ** 2)
        self._add("mae", np.abs(error))
        self._add("mse_raw", raw_error ** 2)
        self._add("mae_raw", np.abs(raw_error))
        self._add("mase", np.abs(raw_error) / mase_scale[None, :, None])
        self.batch_mse.append(float(np.mean(error ** 2)))
        self.batch_mae.append(float(np.mean(np.abs(error))))
        for q, values in forecast.quantiles.items():
            diff = truth - values
            self._add(f"pinball_{q:g}", np.maximum(q * diff, (q-1) * diff))
        if 0.1 in forecast.quantiles and 0.9 in forecast.quantiles:
            lower, upper = forecast.quantiles[0.1], forecast.quantiles[0.9]
            self._add("coverage_80", ((truth >= lower) & (truth <= upper)).astype(float))
            self._add("width_80", upper-lower)
            self._add("crossing_80", (upper < lower).astype(float))
            self._add("interval_score_80", upper-lower + 10*np.maximum(lower-truth, 0)
                      + 10*np.maximum(truth-upper, 0))

    def result(self, legacy=False):
        result = {name: float(np.sum(value) / np.sum(self.counts[name]))
                  if np.sum(self.counts[name]) else float("nan") for name, value in self.sums.items()}
        result["n_values"] = int(np.sum(self.counts["mse"]))
        result["n_mase_values"] = int(np.sum(self.counts["mase"]))
        if legacy:
            result["mse_micro"], result["mae_micro"] = result["mse"], result["mae"]
            result["mse"], result["mae"] = float(np.mean(self.batch_mse)), float(np.mean(self.batch_mae))
        return result

    def by_lead(self):
        rows = []
        for i in range(self.horizon):
            rows.append({"lead": i+1, **{name: value[i]/self.counts[name][i]
                         if self.counts[name][i] else np.nan for name, value in self.sums.items()}})
        return pd.DataFrame(rows)


def prediction_frame(data, origins, truth, forecast):
    windows, channels, horizon = truth.shape
    scale, mean = data.scale[None, :, None], data.mean[None, :, None]
    raw_truth, raw_point = truth * scale + mean, forecast.point * scale + mean
    frame = pd.DataFrame({
        "origin": np.repeat(origins, channels*horizon),
        "channel": np.tile(np.repeat(data.channels, horizon), windows),
        "lead": np.tile(np.arange(1, horizon+1), windows*channels),
        "y": raw_truth.ravel(), "prediction": raw_point.ravel(),
        "y_scaled": truth.ravel(), "prediction_scaled": forecast.point.ravel(),
        "error": (raw_point-raw_truth).ravel(),
    })
    for q, values in forecast.quantiles.items():
        frame[f"q{q:g}"] = (values*scale+mean).ravel()
    if "base" in forecast.components and "correction" in forecast.components:
        # The additive correction is a difference: apply scale, never the channel mean.
        frame["base_prediction_scaled"] = forecast.components["base"].ravel()
        frame["correction_scaled"] = forecast.components["correction"].ravel()
        frame["base_prediction"] = (forecast.components["base"]*scale+mean).ravel()
        frame["correction"] = (forecast.components["correction"]*scale).ravel()
    return frame


def evaluate(model, data, origins, context, horizon, batch_size=16, legacy=False, output=None,
             progress_label=None, progress_callback=None):
    windows = Windows(data.values, origins, context, horizon)
    scores, writer = Scores(horizon), None
    prediction_seconds = 0.0
    progress = Progress(len(windows), progress_label, progress_callback) if progress_label else None
    try:
        for start in range(0, len(windows), batch_size):
            batch = [windows[i] for i in range(start, min(start+batch_size, len(windows)))]
            x, truth = np.stack([b[0] for b in batch]), np.stack([b[1] for b in batch])
            begin = perf_counter()
            forecast = model.predict(x, horizon).validate(truth.shape)
            prediction_seconds += perf_counter() - begin
            scores.update(truth, forecast, data.scale, data.mase_scale)
            if output:
                frame = prediction_frame(data, origins[start:start+len(batch)], truth, forecast)
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    Path(output).parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(output, table.schema, compression="zstd")
                    # A small standalone sample for plotting without loading the full parquet.
                    frame[(frame.origin == origins[0]) & (frame.channel == data.channels[0])].to_csv(
                        Path(output).with_suffix(".example.csv"), index=False)
                writer.write_table(table)
            if progress:
                progress.update(start + len(batch))
    finally:
        if writer is not None:
            writer.close()
    return scores.result(legacy), scores.by_lead(), prediction_seconds
