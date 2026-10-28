"""Check the requested catalog, configuration migration and hybrid dependencies."""
from pathlib import Path

import pytest

from tshybrid.models.registry import FOUNDATION_MODELS, MODEL_TYPES, NEURAL_ADAPTERS, STATISTICAL_MODELS
from tshybrid.runner import read_config, validate_config


def test_requested_models_have_individual_implementation_modules():
    classes = {**STATISTICAL_MODELS, **NEURAL_ADAPTERS, **FOUNDATION_MODELS}
    assert set(classes) == {"ar", "arima", "dlinear", "tsmixer", "moderntcn", "patchtst",
                            "itransformer", "chronos", "timesfm", "moirai"}
    for name, cls in classes.items():
        assert cls.__module__ == f"tshybrid.models.{name}"
    assert MODEL_TYPES == set(classes) | {"hybrid"}


def test_all_shipped_configs_use_supported_models():
    directory = Path(__file__).resolve().parents[1] / "configs"
    for path in directory.glob("*.yaml"):
        validate_config(read_config(path))
    full = read_config(directory / "benchmark.yaml")
    assert {m["type"] for m in full["models"]} == MODEL_TYPES


@pytest.mark.parametrize("kind", ["last_value", "seasonal_naive", "ets", "linear"])
def test_removed_models_are_rejected_before_starting_a_run(kind):
    config = {"purpose": "pilot", "context": 12, "horizons": [4],
              "datasets": [{"name": "test"}], "models": [{"id": "removed", "type": kind}]}
    with pytest.raises(ValueError, match="Unknown model type"):
        validate_config(config)


def test_retained_hybrids_require_causal_statistical_base():
    config = {"purpose": "pilot", "context": 12, "horizons": [4],
              "datasets": [{"name": "test"}],
              "models": [{"id": "base", "type": "dlinear"}, {"id": "fm", "type": "chronos"},
                         {"id": "combined", "type": "hybrid", "base": "base", "residual": "fm"}]}
    with pytest.raises(ValueError, match="Hybrid base must implement causal residuals"):
        validate_config(config)
