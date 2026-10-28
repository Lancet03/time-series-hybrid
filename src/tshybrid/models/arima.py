import warnings
import numpy as np

from .base import Forecast, Model


def _fit_one(history, horizon, order, fit_prefix, maxiter, initialization):
    """Independent CPU task: identical estimation for serial and process backends."""
    from statsmodels.tsa.arima.model import ARIMA as StatsARIMA

    prefix = fit_prefix or len(history)
    if not max(order) + 3 <= prefix <= len(history):
        raise ValueError("Invalid ARIMA fit_prefix")
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always")
        estimator = StatsARIMA(history[:prefix].astype(np.float64), order=order)
        if initialization == "approximate_diffuse":
            estimator.initialize_approximate_diffuse()
        fitted = estimator.fit(method_kwargs={"maxiter": maxiter})
        nonconverged = not fitted.mle_retvals.get("converged", True)
        residuals = None
        if prefix < len(history):
            # Same causal filtering, with unchanged prefix-fitted coefficients.
            fitted = fitted.extend(history[prefix:].astype(np.float64))
            residuals = np.asarray(fitted.resid)
        point = np.asarray(fitted.forecast(horizon))
    counts = {}
    for notice in notices:
        key = f"{notice.category.__name__}: {notice.message}"
        counts[key] = counts.get(key, 0) + 1
    return point, residuals, int(nonconverged), counts


class ARIMA(Model):
    def __init__(self, order=(2, 1, 0), fit_prefix=None, maxiter=100, initialization="default", workers=1):
        if initialization not in {"default", "approximate_diffuse"}:
            raise ValueError("ARIMA initialization must be default or approximate_diffuse")
        if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
            raise ValueError("ARIMA workers must be a positive integer")
        self.order, self.fit_prefix, self.maxiter = tuple(order), fit_prefix, maxiter
        self.initialization, self.workers = initialization, workers
        self.warning_counts = {}
        self.metadata = {"order": list(order), "fit_prefix": fit_prefix, "maxiter": maxiter,
                         "initialization": initialization, "workers": workers,
                         "warning_counts": self.warning_counts, "fit_calls": 0,
                         "refit": "each context; fixed prefix parameters, updated state if fit_prefix set"}
        self.nonconverged_fits = 0

    def _results(self, x, horizon):
        histories = np.asarray(x).reshape(-1, x.shape[-1])
        args = (horizon, self.order, self.fit_prefix, self.maxiter, self.initialization)
        if self.workers == 1:
            results = [_fit_one(history, *args) for history in histories]
        else:
            from joblib import Parallel, delayed, parallel_config
            # loky reuses its process pool between evaluator batches. One BLAS
            # thread per process prevents nested CPU thread oversubscription.
            with parallel_config(backend="loky", n_jobs=self.workers, inner_max_num_threads=1):
                results = Parallel()(delayed(_fit_one)(history, *args) for history in histories)
        for _, _, nonconverged, counts in results:
            self.nonconverged_fits += nonconverged
            self.metadata["fit_calls"] += 1
            for key, count in counts.items():
                self.warning_counts[key] = self.warning_counts.get(key, 0) + count
        return results

    def _one(self, history, horizon):
        point, residuals, _, _ = self._results(np.asarray(history)[None, :], horizon)[0]
        return point, residuals

    def predict(self, x, horizon):
        results = self._results(x, horizon)
        return Forecast(np.stack([row[0] for row in results]).reshape(*x.shape[:-1], horizon))

    def components(self, x, horizon, warmup):
        if self.fit_prefix != warmup or warmup >= x.shape[-1]:
            raise ValueError("Hybrid ARIMA requires fit_prefix == warmup < context")
        results = self._results(x, horizon)
        out = np.stack([row[0] for row in results]).reshape(*x.shape[:-1], horizon)
        residuals = np.stack([row[1] for row in results]).reshape(*x.shape[:-1], x.shape[-1] - warmup)
        return Forecast(out), residuals
