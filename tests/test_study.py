import numpy as np
import pandas as pd
import pytest

from tshybrid.data import load_dataset, make_origins
from tshybrid.evaluation import evaluate
from tshybrid.models.hybrid import Hybrid
from tshybrid.models.statistical import LastValue
from tshybrid.study import gain, paired_predictions


def test_hybrid_components_roundtrip_in_nonzero_original_units(tmp_path):
    values = 500 + np.arange(240).reshape(120, 2).astype(float)
    np.save(tmp_path / "series.npy", values)
    data = load_dataset(dict(name="test", path="series.npy", seasonality=2, split=[.6, .2, .2]), tmp_path)
    origins = make_origins(data.val_end, 120, 12, 4, limit=3)
    for name, model in [("base", LastValue()), ("fm", LastValue()),
                        ("hybrid", Hybrid(LastValue(), LastValue(), warmup=3))]:
        evaluate(model, data, origins, 12, 4, batch_size=2, output=tmp_path / f"{name}.parquet")
    frames = [pd.read_parquet(tmp_path / f"{name}.parquet") for name in ("base", "fm", "hybrid")]
    hybrid = frames[-1]
    np.testing.assert_allclose(hybrid.prediction, hybrid.base_prediction+hybrid.correction)
    # Every raw one-step increment is 2, regardless of the large, nonzero train mean.
    np.testing.assert_allclose(hybrid.correction, 2, atol=1e-5)
    paired = paired_predictions(*frames)
    assert len(paired) == 3*2*4
    assert paired.hybrid_mse.mean() < paired.base_mse.mean()
    with pytest.raises(ValueError, match="Different prediction targets"):
        paired_predictions(frames[0].iloc[:-1], frames[1], frames[2])
    damaged = frames[2].copy()
    damaged["base_prediction_scaled"] += .5
    with pytest.raises(AssertionError, match="different base forecast"):
        paired_predictions(frames[0], frames[1], damaged)


def test_zero_baseline_gain_is_undefined_and_invalid_warmup_rejected():
    assert np.isnan(gain(0, .1))
    assert gain(2, 1) == 50
    for warmup in (0, -1, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            Hybrid(LastValue(), LastValue(), warmup=warmup)


def test_numerically_failed_validation_candidate_is_recorded_and_next_one_runs(tmp_path, monkeypatch):
    import json
    import yaml
    import tshybrid.runner as runner
    np.save(tmp_path / "series.npy", np.arange(240, dtype=float).reshape(120, 2))
    original_factory = runner.make_model
    def factory(spec, params, *args):
        if params.get("fail"):
            raise np.linalg.LinAlgError("Numerical failure in parameter estimation")
        return original_factory(spec, params, *args)
    monkeypatch.setattr(runner, "make_model", factory)
    config = dict(name="numeric", purpose="pilot", context=12, horizons=[4], threads=1,
                  output="out", evaluation=dict(val_limit=2, test_limit=2),
                  datasets=[dict(name="test", path="series.npy", seasonality=2, split=[.6,.2,.2])],
                  models=[dict(id="last", type="last_value", candidates=[dict(fail=True), {}])])
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    directory, success = runner.run(path)
    assert success
    selection = json.loads((directory / "selection.json").read_text())
    assert selection[0]["status"] == "numerical_failure"
    assert "validation" in selection[1]
    assert len(pd.read_csv(directory / "metrics.csv")) == 1
