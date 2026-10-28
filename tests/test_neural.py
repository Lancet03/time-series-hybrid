"""Standalone local-network training and checkpoint checks."""
import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from tshybrid.models.dlinear import DLinearModel as DLinear, DLinear as DLinearNet
from tshybrid.models.registry import NEURAL_MODELS
from tshybrid.runner import run


def test_dlinear_adapter_uses_local_class():
    assert type(DLinear(336, 48, 7).net) is DLinearNet
    assert DLinearNet.__module__ == "tshybrid.models.dlinear"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_all_local_models_train_evaluate_and_save_checkpoints(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware not available")
    np.save(tmp_path / "series.npy", np.random.default_rng(42).normal(size=(900, 3)))
    config = {
        "name": "Local neural integration", "purpose": "pilot", "protocol": "common",
        "context": 336, "horizons": [48], "device": device, "threads": 1,
        "output": "outputs", "evaluation": {"batch_size": 2, "val_limit": 2, "test_limit": 3},
        "datasets": [{"name": "test", "path": "series.npy", "seasonality": 24, "split": [.6, .2, .2]}],
        "models": [{"id": name, "type": name,
                    "params": {"epochs": 1, "batch_size": 4, "train_limit": 8}}
                   for name in NEURAL_MODELS],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output, success = run(path)
    assert success, (output / "failures.json").read_text(encoding="utf-8")
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "completed"
    assert progress["completed_tasks"] == progress["total_tasks"] == len(NEURAL_MODELS)
    assert progress["failed_tasks"] == 0
    assert progress["completed_epochs"] == progress["total_epochs"] == len(NEURAL_MODELS)
    scores = pd.read_csv(output / "metrics.csv")
    assert set(scores.model) == set(NEURAL_MODELS)
    assert np.isfinite(scores[["mse", "mae"]].to_numpy()).all()
    for name in NEURAL_MODELS:
        folder = output / f"test-{name}-H48-s0"
        info = json.loads((folder / "model.json").read_text(encoding="utf-8"))
        assert info["metadata"]["implementation"] == "HN-MVTS"
        assert info["metadata"]["network_class"].startswith("tshybrid.models.")
        network = NEURAL_MODELS[name][0]
        restored = network(d_input=336, d_output=48, n_input=3, **info["metadata"]["model_params"])
        restored.load_state_dict(torch.load(folder / "best.pt", weights_only=True))
        predictions = pd.read_parquet(folder / "predictions.parquet")
        assert len(predictions) == 3 * 3 * 48
        assert predictions.origin.nunique() == 3
