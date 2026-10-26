"""Local models. fit_prefix pins parameter estimation before residual observations."""
import warnings

import numpy as np

from .base import Forecast, Model


class LastValue(Model):
    metadata = {"point_forecast": "last observation", "refit": "each context"}

    def predict(self, x, horizon):
        return Forecast(np.repeat(x[..., -1:], horizon, axis=-1))

    def components(self, x, horizon, warmup):
        return self.predict(x, horizon), x[..., warmup:] - x[..., warmup-1:-1]


class SeasonalNaive(Model):
    def __init__(self, period):
        self.period = period
        self.metadata = {"period": period, "refit": "each context"}

    def predict(self, x, horizon):
        if x.shape[-1] < self.period:
            raise ValueError("Context shorter than seasonal period")
        return Forecast(x[..., -self.period:][..., np.arange(horizon) % self.period])

    def components(self, x, horizon, warmup):
        if warmup < self.period:
            raise ValueError("Residual warmup shorter than seasonality")
        return self.predict(x, horizon), x[..., warmup:] - x[..., warmup-self.period:-self.period]


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


class ARIMA(Model):
    def __init__(self, order=(2, 1, 0), fit_prefix=None, maxiter=100, initialization="default"):
        if initialization not in {"default", "approximate_diffuse"}:
            raise ValueError("ARIMA initialization must be default or approximate_diffuse")
        self.order, self.fit_prefix, self.maxiter = tuple(order), fit_prefix, maxiter
        self.initialization = initialization
        self.metadata = {"order": list(order), "fit_prefix": fit_prefix, "maxiter": maxiter,
                         "initialization": initialization,
                         "refit": "each context; fixed prefix parameters, updated state if fit_prefix set"}
        self.nonconverged_fits = 0

    def _one(self, history, horizon):
        from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        prefix = self.fit_prefix or len(history)
        if not max(self.order) + 3 <= prefix <= len(history):
            raise ValueError("Invalid ARIMA fit_prefix")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            estimator = StatsARIMA(history[:prefix].astype(np.float64), order=self.order)
            if self.initialization == "approximate_diffuse":
                estimator.initialize_approximate_diffuse()
            fitted = estimator.fit(method_kwargs={"maxiter": self.maxiter})
        if not fitted.mle_retvals.get("converged", True):
            self.nonconverged_fits += 1
        residuals = None
        if prefix < len(history):
            # extend runs a causal Kalman filter with unchanged parameters, no smoothing/refit.
            fitted = fitted.extend(history[prefix:].astype(np.float64))
            residuals = np.asarray(fitted.resid)
        return np.asarray(fitted.forecast(horizon)), residuals

    def predict(self, x, horizon):
        out = np.empty((*x.shape[:-1], horizon))
        for index in np.ndindex(x.shape[:-1]):
            out[index], _ = self._one(x[index], horizon)
        return Forecast(out)

    def components(self, x, horizon, warmup):
        if self.fit_prefix != warmup or warmup >= x.shape[-1]:
            raise ValueError("Hybrid ARIMA requires fit_prefix == warmup < context")
        out = np.empty((*x.shape[:-1], horizon))
        residuals = np.empty((*x.shape[:-1], x.shape[-1] - warmup))
        for index in np.ndindex(x.shape[:-1]):
            out[index], residuals[index] = self._one(x[index], horizon)
        return Forecast(out), residuals


class ETS(Model):
    def __init__(self, trend="add", seasonal=None, period=None):
        self.params = {"trend": trend, "seasonal": seasonal,
                       "seasonal_periods": period if seasonal else None}
        self.metadata = {**self.params, "refit": "each context"}

    def predict(self, x, horizon):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        out = np.empty((*x.shape[:-1], horizon))
        for index in np.ndindex(x.shape[:-1]):
            fitted = ExponentialSmoothing(x[index].astype(np.float64), **self.params,
                                          initialization_method="estimated").fit(optimized=True)
            out[index] = fitted.forecast(horizon)
        return Forecast(out)
