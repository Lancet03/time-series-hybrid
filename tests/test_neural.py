"""Numerical reference comparisons and complete local-network training checks."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from tshybrid.models.neural import DLinear, DLinearNet, NEURAL_MODELS
from tshybrid.runner import run


REFERENCE_NAMES = {
    "dlinear": ("dlinear", "DLinear"),
    "tsmixer": ("tsmixer", "TSMixer"),
    "moderntcn": ("moderntcn", "ModernTCN"),
    "patchtst": ("patchtst", "PatchTST"),
    "itransformer": ("itransformer", "InverseTransformer"),
}


@pytest.mark.parametrize("architecture", NEURAL_MODELS)
def test_local_network_matches_reference_forward_and_gradients(architecture, monkeypatch):
    root = Path(__file__).resolve().parents[2] / "HN-MVTS" / "src"
    if not root.exists():
        pytest.skip("Optional external reference checkout unavailable")

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    monkeypatch.setitem(sys.modules, "layers", load("reference_layers", root / "layers.py"))
    filename, class_name = REFERENCE_NAMES[architecture]
    reference_class = getattr(load(f"reference_{filename}", root / "models" / f"{filename}.py"), class_name)
    network, defaults, _, _ = NEURAL_MODELS[architecture]
    # Exercise patching and downsampling with both even and odd patch counts.
    context = 340 if architecture in {"moderntcn", "patchtst"} else 48
    params = dict(d_input=context, d_output=12, n_input=3, **defaults)
    torch.manual_seed(11)
    reference = reference_class(**params)
    torch.manual_seed(11)
    actual = network(**params)
    assert type(actual).__module__.startswith("tshybrid.models.")
    # Preserve initialization order and checkpoint names, not just the forward formula.
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)
    actual.load_state_dict(reference.state_dict(), strict=True)
    reference.eval()
    actual.eval()
    x = torch.randn(2, 3, context, requires_grad=True)
    other = x.detach().clone().requires_grad_(True)
    expected, result = reference(x), actual(other)
    torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)
    expected.square().mean().backward()
    result.square().mean().backward()
    torch.testing.assert_close(other.grad, x.grad, rtol=1e-5, atol=1e-6)
    for name, parameter in reference.named_parameters():
        local = dict(actual.named_parameters())[name]
        if parameter.grad is None:
            assert local.grad is None
        else:
            torch.testing.assert_close(local.grad, parameter.grad, rtol=1e-5, atol=1e-6)


def test_dlinear_adapter_uses_local_class():
    assert type(DLinear(336, 48, 7).net) is DLinearNet
    assert DLinearNet.__module__ == "tshybrid.models.networks"


def test_all_local_models_train_evaluate_and_save_checkpoints(tmp_path):
    np.save(tmp_path / "series.npy", np.random.default_rng(42).normal(size=(900, 3)))
    config = {
        "name": "Local neural integration", "purpose": "pilot", "protocol": "common",
        "context": 336, "horizons": [48], "device": "cpu", "threads": 1,
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
    scores = pd.read_csv(output / "metrics.csv")
    assert set(scores.model) == set(NEURAL_MODELS)
    assert np.isfinite(scores[["mse", "mae"]].to_numpy()).all()
    for name in NEURAL_MODELS:
        folder = output / f"test-{name}-H48-s0"
        info = json.loads((folder / "model.json").read_text(encoding="utf-8"))
        assert info["metadata"]["implementation"] == "time-series-hybrid"
        assert info["metadata"]["network_class"].startswith("tshybrid.models.")
        network = NEURAL_MODELS[name][0]
        restored = network(d_input=336, d_output=48, n_input=3, **info["metadata"]["model_params"])
        restored.load_state_dict(torch.load(folder / "best.pt", weights_only=True))
        predictions = pd.read_parquet(folder / "predictions.parquet")
        assert len(predictions) == 3 * 3 * 48
        assert predictions.origin.nunique() == 3
