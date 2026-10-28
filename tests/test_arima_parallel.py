"""Parallel CPU fitting must preserve forecasts, causal residuals and diagnostics."""
import warnings

import numpy as np
import pytest
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from tshybrid.models.arima import ARIMA


@pytest.mark.parametrize("order,initialization", [((2, 1, 0), "default"),
                                                ((2, 0, 0), "approximate_diffuse")])
def test_parallel_arima_matches_serial_forecasts_and_residuals(order, initialization):
    x = np.random.default_rng(42).normal(size=(3, 2, 64)).astype(np.float32)
    params = dict(order=order, fit_prefix=32, maxiter=30, initialization=initialization)
    serial, parallel = ARIMA(**params), ARIMA(**params, workers=2)
    expected, expected_residuals = serial.components(x, 8, 32)
    actual, residuals = parallel.components(x, 8, 32)
    np.testing.assert_allclose(actual.point, expected.point, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(residuals, expected_residuals, rtol=1e-10, atol=1e-10)
    assert parallel.nonconverged_fits == serial.nonconverged_fits
    assert parallel.metadata["fit_calls"] == serial.metadata["fit_calls"] == 6
    assert parallel.warning_counts == serial.warning_counts
    # Another call uses the reusable worker pool, without stale results or reordered channels.
    np.testing.assert_allclose(parallel.predict(x[::-1].copy(), 8).point, expected.point[::-1],
                               rtol=1e-10, atol=1e-10)


def test_arima_records_convergence_warnings_without_flooding_console():
    x = np.random.default_rng(7).normal(size=(2, 2, 48)).astype(np.float32)
    for workers in (1, 2):
        model = ARIMA(order=(2, 1, 0), fit_prefix=24, maxiter=1, workers=workers)
        with warnings.catch_warnings(record=True) as leaked:
            warnings.simplefilter("always")
            model.predict(x, 4)
        # Dependency-import deprecations may appear once on the first call;
        # per-fit numerical warnings must stay in model diagnostics.
        assert not any(issubclass(w.category, ConvergenceWarning)
                       or "starting autoregressive" in str(w.message) for w in leaked)
        assert model.nonconverged_fits > 0
        assert any("ConvergenceWarning" in key for key in model.warning_counts)


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_arima_rejects_invalid_worker_counts(workers):
    with pytest.raises(ValueError, match="workers must be a positive integer"):
        ARIMA(workers=workers)
