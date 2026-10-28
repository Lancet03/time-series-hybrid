"""Deterministic forecast double for evaluator tests; not a registered model."""
import numpy as np

from tshybrid.models.base import Forecast, Model


class Persistence(Model):
    def predict(self, x, horizon):
        return Forecast(np.repeat(x[..., -1:], horizon, axis=-1))

    def components(self, x, horizon, warmup):
        return self.predict(x, horizon), x[..., warmup:] - x[..., warmup-1:-1]
