"""Frozen Chronos-2, evaluating each channel independently without cross-learning."""
import numpy as np
import torch

from .foundation import FrozenFoundation


class Chronos2(FrozenFoundation):
    family = "chronos2"
    package = "chronos"

    def load(self):
        from chronos import Chronos2Pipeline

        self.pipeline = Chronos2Pipeline.from_pretrained(
            self.path, device_map=self.device, torch_dtype=torch.float32,
        )
        self.pipeline.model.eval().requires_grad_(False)
        self.metadata.update({
            "point_forecast": "median (q0.5)",
            "native_horizon": self.pipeline.model_prediction_length,
            "forecast_flags": {"cross_learning": False},
        })

    def predict_batch(self, batch, horizon):
        quantiles, points = self.pipeline.predict_quantiles(
            # A list of 1D arrays denotes separate univariate tasks, not joint targets.
            [np.asarray(row, dtype=np.float32) for row in batch],
            prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9],
            batch_size=self.batch_size, context_length=batch.shape[-1],
            cross_learning=False, limit_prediction_length=True,
        )
        return (
            torch.cat(points, dim=0).cpu().numpy(),
            torch.cat(quantiles, dim=0).cpu().numpy(),
        )
