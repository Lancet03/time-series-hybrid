import warnings
import numpy as np

from .base import Forecast, Model


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
