import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tshybrid.data import Windows, load_dataset, make_origins
from tshybrid.evaluation import Scores, evaluate
from tshybrid.models.base import Forecast
from tshybrid.models.hybrid import Hybrid
from tshybrid.models.neural import DLinear, DLinearNet
from tshybrid.models.statistical import AR, ARIMA, LastValue
from tshybrid.reporting import combine_runs, compare_references
from tshybrid.runner import read_config, run


def dataset(tmp_path, values=None):
    values = np.arange(240, dtype=float).reshape(120, 2) if values is None else values
    np.save(tmp_path / "series.npy", values)
    return load_dataset({"name": "test", "path": "series.npy", "seasonality": 2,
                         "split": [.6, .2, .2]}, tmp_path)


def test_full_window_boundaries_and_train_only_scaler(tmp_path):
    original = np.arange(240, dtype=float).reshape(120, 2)
    first = dataset(tmp_path, original)
    changed = original.copy()
    changed[first.train_end:] += 10000
    second = dataset(tmp_path, changed)
    np.testing.assert_equal(first.mean, second.mean)
    np.testing.assert_equal(first.scale, second.scale)
    origins = make_origins(first.val_end, 120, 12, 6)
    assert origins[0] == first.val_end
    assert origins[-1] + 6 == 120
    windows = Windows(first.values, origins, 12, 6)
    x, y = windows[0]
    np.testing.assert_equal(x, first.values[first.val_end-12:first.val_end].T)
    np.testing.assert_equal(y, first.values[first.val_end:first.val_end+6].T)


def test_legacy_boundaries_are_explicit():
    assert make_origins(100, 160, 12, 6)[-1] == 154
    assert make_origins(100, 160, 12, 6, legacy=True)[-1] == 154
    with pytest.raises(ValueError):
        make_origins(3, 10, 12, 6)


def test_hn_window_start_indices_convert_to_full_target_origins():
    # Literal author slicing, independently represented by integer-valued timestamps.
    timestamps = np.arange(200)[:, None]
    context, horizon, start, end = 12, 6, 100, 160
    xy = np.lib.stride_tricks.sliding_window_view(timestamps, context+horizon, axis=0)
    author_windows = xy[start-context:end-context-horizon+1]
    author_origins = author_windows[:, 0, context]
    np.testing.assert_array_equal(author_origins, make_origins(start, end, context, horizon, legacy=True))
    assert author_origins[-1]+horizon == end


def test_metrics_weight_short_final_batch_and_zero_mase():
    scores = Scores(1)
    scores.update(np.zeros((2, 1, 1)), Forecast(np.ones((2, 1, 1))), np.ones(1), np.array([np.nan]))
    scores.update(np.zeros((1, 1, 1)), Forecast(np.full((1, 1, 1), 3)), np.ones(1), np.array([np.nan]))
    assert scores.result()["mse"] == pytest.approx(11/3)
    assert scores.result(legacy=True)["mse"] == 5
    assert scores.result()["n_mase_values"] == 0
    assert np.isnan(scores.result()["mase"])


def test_interval_scores():
    scores = Scores(1)
    truth = np.array([[[0.]], [[3.]]])
    forecast = Forecast(np.ones_like(truth), {0.1: -np.ones_like(truth), 0.9: np.ones_like(truth)})
    scores.update(truth, forecast, np.ones(1), np.ones(1))
    assert scores.result()["coverage_80"] == .5
    assert scores.result()["interval_score_80"] == 12


@pytest.mark.parametrize("factory", [lambda: AR(order=3, fit_prefix=24),
                                    lambda: ARIMA(order=(1, 0, 0), fit_prefix=24),
                                    lambda: ARIMA(order=(1, 0, 0), fit_prefix=24, initialization="approximate_diffuse")])
def test_residuals_are_causal_and_base_matches(factory):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(1, 1, 48)).astype(np.float32)
    model = factory()
    base, residuals = model.components(x, 4, 24)
    np.testing.assert_allclose(base.point, model.predict(x, 4).point)
    changed = x.copy()
    changed[..., 39:] += 100
    _, new_residuals = model.components(changed, 4, 24)
    np.testing.assert_allclose(residuals[..., :15], new_residuals[..., :15])


def test_hybrid_does_not_receive_future_targets(tmp_path):
    data = dataset(tmp_path)
    model = Hybrid(AR(order=3, fit_prefix=12), LastValue(), warmup=12)
    origin, context, horizon = data.val_end, 24, 4
    x = data.values[origin-context:origin].T[None]
    before = model.predict(x, horizon).point.copy()
    data.values[origin:] += 10000
    after = model.predict(data.values[origin-context:origin].T[None], horizon).point
    np.testing.assert_array_equal(before, after)


def test_predictions_roundtrip_and_mae_matches_saved_values(tmp_path):
    data = dataset(tmp_path)
    origins = make_origins(data.val_end, len(data.values), 12, 4, limit=3)
    output = tmp_path / "predictions.parquet"
    metrics, leads, _ = evaluate(LastValue(), data, origins, 12, 4, batch_size=2, output=output)
    saved = pd.read_parquet(output)
    assert len(saved) == 3 * 2 * 4
    assert metrics["mae_raw"] == pytest.approx(np.abs(saved.prediction-saved.y).mean())
    assert metrics["mae"] == pytest.approx(np.abs(saved.prediction_scaled-saved.y_scaled).mean())
    assert list(leads.lead) == [1, 2, 3, 4]


def test_dlinear_training_never_reads_test(tmp_path):
    data = dataset(tmp_path, np.random.default_rng(3).normal(size=(120, 2)))
    origins = make_origins(data.train_end, data.val_end, 12, 4, limit=3)
    first = DLinear(12, 4, 2, seed=2, epochs=2, batch_size=8, train_limit=16)
    first.fit(data, origins, "common")
    expected = {k: v.clone() for k, v in first.net.state_dict().items()}
    data.values[data.val_end:] = 100000
    second = DLinear(12, 4, 2, seed=2, epochs=2, batch_size=8, train_limit=16)
    second.fit(data, origins, "common")
    for key, value in second.net.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_dlinear_matches_hn_source_when_available():
    import importlib.util
    import sys
    root = Path(__file__).resolve().parents[2] / "HN-MVTS" / "src"
    if not root.exists():
        pytest.skip("External HN-MVTS checkout unavailable")
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("hn_reference_dlinear", root / "models" / "dlinear.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(11)
    reference = module.DLinear(48, 12, 3)
    torch.manual_seed(11)
    actual = DLinearNet(48, 12, 3)
    x = torch.randn(2, 3, 48)
    torch.testing.assert_close(actual(x), reference(x), rtol=1e-6, atol=1e-6)
    sys.path.remove(str(root))


def test_reference_comparison_rejects_pilot(tmp_path):
    reference = pd.DataFrame([dict(dataset="x", model="dlinear", horizon=4, protocol="hn_mvts_code",
                                   expected_mse=.3, tolerance=.01)])
    reference.to_csv(tmp_path / "reference.csv", index=False)
    metrics = pd.DataFrame([dict(dataset="x", model="dlinear", horizon=4, protocol="hn_mvts_code",
                                 mse=.3, pilot=True)])
    assert compare_references(metrics, tmp_path / "reference.csv").iloc[0].status == "protocol_mismatch"
    reference["steps"], reference["val_end"] = 200, 160
    reference.to_csv(tmp_path / "reference.csv", index=False)
    metrics["steps"], metrics["val_end"] = 200, 160
    metrics["pilot"], metrics["n_origins"] = False, 25
    result = compare_references(metrics, tmp_path / "reference.csv").iloc[0]
    assert result.status == "protocol_mismatch" and "n_origins" in result.mismatches
    metrics["n_origins"] = 37
    assert compare_references(metrics, tmp_path / "reference.csv").iloc[0].status == "within_tolerance"


def test_config_inheritance_and_cycles(tmp_path):
    (tmp_path / "base.yaml").write_text("context: 12\nevaluation:\n  batch_size: 4\n  test_limit: 2\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nevaluation:\n  batch_size: 8\n")
    assert read_config(tmp_path / "child.yaml")["evaluation"] == {"batch_size": 8, "test_limit": 2}
    (tmp_path / "cycle.yaml").write_text("extends: cycle.yaml\n")
    with pytest.raises(ValueError):
        read_config(tmp_path / "cycle.yaml")


def test_failed_models_are_visible_and_successful_results_survive(tmp_path):
    import yaml
    dataset(tmp_path)
    config = dict(name="integration", purpose="pilot", context=12, horizons=[4], threads=1,
                  output="out", evaluation=dict(val_limit=2, test_limit=2),
                  datasets=[dict(name="test", path="series.npy", seasonality=2, split=[.6,.2,.2])],
                  models=[dict(id="last", type="last_value"), dict(id="broken", type="unknown")])
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    output, success = run(path)
    assert not success
    assert json.loads((output / "manifest.json").read_text())["status"] == "completed_with_failures"
    assert len(json.loads((output / "failures.json").read_text())) == 1
    assert len(pd.read_csv(output / "metrics.csv")) == 1
    assert (output / "report.html").exists()


def test_combining_runs_checks_data_and_preserves_source_ids(tmp_path):
    import yaml
    dataset(tmp_path)
    config = dict(name="combine", purpose="pilot", context=12, horizons=[4], threads=1,
                  output="out", evaluation=dict(val_limit=2, test_limit=2),
                  datasets=[dict(name="test", path="series.npy", seasonality=2, split=[.6,.2,.2])],
                  models=[dict(id="last", type="last_value")])
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    first, _ = run(path)
    config["models"] = [dict(id="seasonal", type="seasonal_naive")]
    path.write_text(yaml.safe_dump(config))
    second, _ = run(path)
    combine_runs([first, second], tmp_path / "combined")
    combined = pd.read_csv(tmp_path / "combined" / "metrics.csv")
    assert len(combined) == 2 and combined.run_id.nunique() == 2
    changed = np.load(tmp_path / "series.npy")
    changed[-1] += 10
    np.save(tmp_path / "series.npy", changed)
    third, _ = run(path)
    with pytest.raises(ValueError, match="Different data/protocol/origins"):
        combine_runs([first, third], tmp_path / "rejected")
