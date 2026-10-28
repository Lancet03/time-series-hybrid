import numpy as np

from .base import Forecast, Model


class AR(Model):
    def __init__(self, order=24, ridge=1e-6, fit_prefix=None):
        self.order, self.ridge, self.fit_prefix = order, ridge, fit_prefix
        self.metadata = {"order": order, "ridge": ridge, "fit_prefix": fit_prefix,
                         "refit": "each context; coefficients frozen after fit_prefix if set"}

    def _coefficients(self, x, prefix):
        if not self.order + 2 <= prefix <= x.shape[-1]:
            raise ValueError("AR needs prefix >= order+2 and <= context")
        history = x[..., :prefix].astype(np.float64)
        windows = np.lib.stride_tricks.sliding_window_view(history, self.order + 1, axis=-1)
        features = np.concatenate([np.ones((*windows.shape[:-1], 1)), windows[..., :-1]], axis=-1)
        target = windows[..., -1]
        xt = features.swapaxes(-1, -2)
        penalty = self.ridge * np.eye(self.order + 1)
        penalty[0, 0] = 0
        # Batched solves share the same implementation for every channel and window.
        return np.linalg.solve(xt @ features + penalty, (xt @ target[..., None]))[..., 0]

    def _forecast(self, x, horizon, coefficients):
        history = np.concatenate([x.astype(np.float64), np.zeros((*x.shape[:-1], horizon))], axis=-1)
        length = x.shape[-1]
        for lead in range(horizon):
            history[..., length+lead] = coefficients[..., 0] + np.sum(
                coefficients[..., 1:] * history[..., length+lead-self.order:length+lead], axis=-1)
        return Forecast(history[..., -horizon:])

    def predict(self, x, horizon):
        coefficients = self._coefficients(x, self.fit_prefix or x.shape[-1])
        return self._forecast(x, horizon, coefficients)

    def components(self, x, horizon, warmup):
        if self.fit_prefix != warmup:
            raise ValueError("Hybrid AR requires fit_prefix == residual warmup")
        coefficients = self._coefficients(x, warmup)
        windows = np.lib.stride_tricks.sliding_window_view(x, self.order, axis=-1)
        lagged = windows[..., warmup-self.order:-1, :]
        one_step = coefficients[..., 0, None] + np.sum(coefficients[..., None, 1:] * lagged, axis=-1)
        return self._forecast(x, horizon, coefficients), x[..., warmup:] - one_step
