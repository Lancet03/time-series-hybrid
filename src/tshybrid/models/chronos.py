"""Frozen Chronos-Bolt adapter using the official chronos-forecasting API."""
import torch

from .foundation import FrozenFoundation


class Chronos(FrozenFoundation):
    family = "chronos"
    package = "chronos"

    def load(self):
        from chronos import BaseChronosPipeline
        self.pipeline = BaseChronosPipeline.from_pretrained(self.path, device_map=self.device,
                                                           torch_dtype=torch.float32)
        self.pipeline.model.eval().requires_grad_(False)
        self.metadata["point_forecast"] = "median (Chronos-Bolt API mean is the median)"
        self.metadata["native_horizon"] = self.pipeline.model.chronos_config.prediction_length
        self.metadata["long_horizon_policy"] = "allow upstream autoregressive quantile rollout"

    def predict_batch(self, batch, horizon):
        quantiles, point = self.pipeline.predict_quantiles(
            torch.tensor(batch, dtype=torch.float32), prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9], limit_prediction_length=False)
        return point.cpu().numpy(), quantiles.cpu().numpy()
