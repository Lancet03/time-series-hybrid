"""Frozen TimesFM 3 with independent channels and a median point forecast."""
import numpy as np

from .foundation import FrozenFoundation


class TimesFM3(FrozenFoundation):
    family = "timesfm3"
    package = "timesfm3"

    def load(self):
        from timesfm3 import ModelConfig, TimesFM3Evaluator

        self.pipeline = TimesFM3Evaluator(ModelConfig(
            checkpoint_path=self.path, per_core_batch_size=self.batch_size, device=self.device,
        ))
        self.pipeline.model.eval().requires_grad_(False)
        self.quantile_indices = [self.pipeline.config.quantiles.index(q) for q in [0.1, 0.5, 0.9]]
        self.metadata.update({
            "point_forecast": "median (q0.5)",
            "forecast_flags": {
                "use_symmetric_averaging": True, "make_positive": False,
                "sort_quantiles": True, "use_znorm": False, "padding_mode": "none",
            },
        })

    def predict_batch(self, batch, horizon):
        outputs = list(self.pipeline.predict_batch(
            contexts=[np.asarray(row, dtype=np.float32) for row in batch],
            horizon=horizon, return_quantiles=True,
            # Standardized targets may be negative: never clamp them to positive values.
            **self.metadata["forecast_flags"],
        ))
        if len(outputs) != len(batch):
            raise ValueError("TimesFM 3 returned an unexpected number of forecasts")
        quantiles = np.stack([out.quantiles for out in outputs])[..., self.quantile_indices]
        return quantiles[..., 1], quantiles
