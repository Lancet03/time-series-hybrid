"""Adapter contracts: task isolation, quantile axes and versioned model locking."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tshybrid.cli import main
from tshybrid.models.chronos2 import Chronos2
from tshybrid.models.timesfm3 import TimesFM3
from tshybrid.models.foundation import resolve_revision
from tshybrid.runner import read_config


def test_chronos2_keeps_windows_and_channels_separate():
    adapter = Chronos2.__new__(Chronos2)
    adapter.batch_size = 3
    calls = []

    def predict_quantiles(inputs, **kwargs):
        calls.append(kwargs)
        assert all(row.ndim == 1 for row in inputs)
        horizon = kwargs['prediction_length']
        points = [torch.full((1, horizon), float(row[-1])) for row in inputs]
        quantiles = [torch.stack([point - 1, point, point + 2], dim=-1) for point in points]
        return quantiles, points

    adapter.pipeline = SimpleNamespace(predict_quantiles=predict_quantiles)
    x = np.arange(2 * 4 * 10, dtype=np.float32).reshape(2, 4, 10)
    result = adapter.predict(x, 5).validate((2, 4, 5))
    np.testing.assert_array_equal(result.point, np.repeat(x[..., -1:], 5, axis=-1))
    np.testing.assert_array_equal(result.quantiles[0.1], result.point - 1)
    np.testing.assert_array_equal(result.quantiles[0.9], result.point + 2)
    assert len(calls) == 3  # Includes an incomplete last batch.
    assert all(c['cross_learning'] is False and c['context_length'] == 10 for c in calls)


def test_timesfm3_selects_quantiles_and_preserves_negative_values():
    adapter = TimesFM3.__new__(TimesFM3)
    adapter.batch_size = 2
    adapter.quantile_indices = [0, 4, 8]
    adapter.metadata = {'forecast_flags': {
        'use_symmetric_averaging': True, 'make_positive': False,
        'sort_quantiles': True, 'use_znorm': False, 'padding_mode': 'none',
    }}

    def predict_batch(contexts, horizon, **kwargs):
        assert kwargs['make_positive'] is False
        assert all(row.ndim == 1 for row in contexts)
        for row in contexts:
            quantiles = np.tile(row[-1] + np.arange(9), (horizon, 1))
            # Deliberately different: the adapter must use q0.5 for its point forecast.
            yield SimpleNamespace(quantiles=quantiles, forecast=np.zeros(horizon))

    adapter.pipeline = SimpleNamespace(predict_batch=predict_batch)
    x = -np.arange(1, 61, dtype=np.float32).reshape(2, 3, 10)
    result = adapter.predict(x, 7).validate((2, 3, 7))
    expected = np.repeat(x[..., -1:] + 4, 7, axis=-1)
    np.testing.assert_array_equal(result.point, expected)
    np.testing.assert_array_equal(result.quantiles[0.1], expected - 4)
    np.testing.assert_array_equal(result.quantiles[0.9], expected + 4)


def test_new_models_participate_in_locking_and_prefetch(tmp_path, monkeypatch):
    import huggingface_hub
    import tshybrid.models.foundation as foundation

    configs = Path(__file__).resolve().parents[1] / 'configs'
    cfg = read_config(configs / 'foundation_latest_ettm1.yaml')
    cfg['model_lock'] = 'locked.json'
    config = tmp_path / 'latest.yaml'
    import yaml
    config.write_text(yaml.safe_dump(cfg), encoding='utf-8')

    class API:
        def model_info(self, checkpoint, revision):
            assert revision == 'main'
            return SimpleNamespace(sha='a' * 40)

    monkeypatch.setattr(huggingface_hub, 'HfApi', API)
    main(['lock-models', str(config), '--output', str(tmp_path / 'locked.json')])
    lock = json.loads((tmp_path / 'locked.json').read_text())
    assert set(lock) == {'amazon/chronos-2', 'google/timesfm-3.0-pytorch'}
    calls = []

    def resolve(checkpoint, revision, cache):
        calls.append((checkpoint, revision))
        return '/fake/snapshot', revision

    monkeypatch.setattr(foundation, 'resolve_revision', resolve)
    main(['prefetch', str(config)])
    assert calls == [(spec['params']['checkpoint'], 'a' * 40) for spec in cfg['models']]


@pytest.mark.parametrize('complete', [False, True])
def test_checkpoint_cache_resumes_missing_weights(tmp_path, monkeypatch, complete):
    import huggingface_hub

    snapshot = tmp_path / ('a' * 40)
    snapshot.mkdir()
    (snapshot / 'config.json').write_text('{}')
    if complete:
        (snapshot / 'model.safetensors').write_bytes(b'weights')
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        if not kwargs.get('local_files_only'):
            (snapshot / 'model.safetensors').write_bytes(b'weights')
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, 'snapshot_download', download)
    assert resolve_revision('test/model', 'a' * 40, str(tmp_path)) == (str(snapshot), 'a' * 40)
    assert len(calls) == (1 if complete else 2)
