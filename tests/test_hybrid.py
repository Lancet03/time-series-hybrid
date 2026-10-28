import numpy as np
import pandas as pd
import pytest

from tshybrid.data import load_dataset, make_origins
from tshybrid.evaluation import evaluate
from tshybrid.models.hybrid import Hybrid
from helpers import Persistence


def test_hybrid_components_roundtrip_in_nonzero_original_units(tmp_path):
    values = 500 + np.arange(240).reshape(120, 2).astype(float)
    np.save(tmp_path / "series.npy", values)
    data = load_dataset(dict(name="test", path="series.npy", seasonality=2, split=[.6, .2, .2]), tmp_path)
    origins = make_origins(data.val_end, 120, 12, 4, limit=3)
    for name, model in [("base", Persistence()), ("fm", Persistence()),
                        ("hybrid", Hybrid(Persistence(), Persistence(), warmup=3))]:
        evaluate(model, data, origins, 12, 4, batch_size=2, output=tmp_path / f"{name}.parquet")
    frames = [pd.read_parquet(tmp_path / f"{name}.parquet") for name in ("base", "fm", "hybrid")]
    hybrid = frames[-1]
    np.testing.assert_allclose(hybrid.prediction, hybrid.base_prediction+hybrid.correction)
    # Every raw one-step increment is 2, regardless of the large, nonzero train mean.
    np.testing.assert_allclose(hybrid.correction, 2, atol=1e-5)


def test_invalid_hybrid_warmup_rejected():
    for warmup in (0, -1, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            Hybrid(Persistence(), Persistence(), warmup=warmup)
