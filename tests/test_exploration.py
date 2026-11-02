import numpy as np
import pandas as pd
import pytest
import yaml

from tshybrid.exploration import (
    arima_train_diagnostics, autocorrelations, channel_profile, load_catalog,
    naive_validation, split_arrays, split_drift, train_reference,
)


def item(values):
    return dict(raw=values, train_end=80, val_end=120, channels=["0"],
                spec={"seasonality": 4})


def test_catalog_uses_benchmark_truncation_channels_and_boundaries(tmp_path):
    np.save(tmp_path / "x.npy", np.arange(300).reshape(100, 3))
    config = {"datasets": [dict(name="x", path="x.npy", seasonality=4,
                                split=[.6, .2, .2], take_n=80, channels=[2, 0])]}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    _, catalog = load_catalog(path)
    data = catalog["x"]
    assert data["original_shape"] == (100, 3)
    assert data["raw"].shape == (80, 2)
    assert (data["train_end"], data["val_end"]) == (48, 64)
    assert data["channels"] == ["2", "0"]
    np.testing.assert_array_equal(data["raw"][0], [2, 0])


def test_reference_and_default_profiles_do_not_read_test():
    first = item(np.arange(160, dtype=float)[:, None])
    second = item(first["raw"].copy())
    second["raw"][120:] = np.nan
    for left, right in zip(train_reference(first), train_reference(second)):
        np.testing.assert_array_equal(left, right)
    pd.testing.assert_frame_equal(channel_profile(first), channel_profile(second))
    pd.testing.assert_frame_equal(split_drift(first), split_drift(second))
    assert set(split_arrays(first)) == {"train", "validation"}
    assert "test (post-hoc)" in split_arrays(first, include_test=True)


def test_quality_reports_nonfinite_without_treating_them_as_repeats():
    values = np.tile([0., 1., 2., 3.], 40)[:, None]
    values[90:92] = np.nan
    profile = channel_profile(item(values)).set_index("split")
    assert profile.loc["validation", "nonfinite_pct"] == 5
    assert profile.loc["validation", "repeat_pct"] == 0
    assert not profile.loc["train", "constant"]


def test_seasonal_naive_cycles_last_observed_period_for_long_horizons():
    data = item(np.tile([0., 1., 2., 3.], 40)[:, None])
    scores = naive_validation(data, horizons=[9])
    assert scores.loc[scores.model == "seasonal_naive", "mse"].item() == 0
    persistence = scores[scores.model == "persistence"].iloc[0]
    origins = np.arange(80, 120-9+1, 9)
    expected = np.mean([(data["raw"][o:o+9]-data["raw"][o-1])**2 for o in origins])
    assert persistence.mse == pytest.approx(expected / data["raw"][:80].std()**2)
    data["raw"][120:] = 1e9
    pd.testing.assert_frame_equal(scores, naive_validation(data, horizons=[9]))


def test_acf_does_not_compress_gaps_or_return_spurious_constant_correlations():
    values = np.column_stack([np.arange(20, dtype=float), np.ones(20)])
    values[8, 0] = np.nan
    result = autocorrelations(values, ["gap", "constant"], [1, 4])
    assert result.acf.isna().all()


def test_arima_diagnostics_holdouts_and_no_validation_inputs():
    rng = np.random.default_rng(11)
    data = item(rng.normal(size=(160, 1)))
    before = arima_train_diagnostics(data, context=40, warmup=20, horizon=8,
                                     n_origins=2, order=(1, 0, 0))
    assert before["fits"].origin.max()+8 <= data["train_end"]
    assert before["example"].lead.tolist() == list(range(1, 9))
    data["raw"][80:] += 1e6
    after = arima_train_diagnostics(data, context=40, warmup=20, horizon=8,
                                    n_origins=2, order=(1, 0, 0))
    for key in before:
        pd.testing.assert_frame_equal(before[key], after[key])
