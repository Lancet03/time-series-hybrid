"""Frozen Moirai 1.1 adapter using the official Uni2TS API."""
import numpy as np

from .foundation import FrozenFoundation


class Moirai(FrozenFoundation):
    family = "moirai"
    package = "uni2ts"

    def __init__(self, checkpoint, revision, device="cpu", cache_dir=None,
                 batch_size=16, num_samples=100, patch_size=32):
        self.num_samples, self.patch_size = num_samples, patch_size
        super().__init__(checkpoint, revision, device, cache_dir, batch_size)

    def load(self):
        from uni2ts.model.moirai import MoiraiModule
        self.module = MoiraiModule.from_pretrained(self.path).to(self.device).eval().requires_grad_(False)
        self.metadata.update(point_forecast="sample mean", num_samples=self.num_samples, patch_size=self.patch_size)

    def predict_batch(self, batch, horizon):
        import pandas as pd
        from gluonts.dataset.common import ListDataset
        from uni2ts.model.moirai import MoiraiForecast
        model = MoiraiForecast(module=self.module, prediction_length=horizon,
                               context_length=batch.shape[-1], patch_size=self.patch_size,
                               num_samples=self.num_samples, target_dim=1,
                               feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        predictor = model.create_predictor(batch_size=self.batch_size, device=self.device)
        dataset = ListDataset([{"start": pd.Period("2000-01-01", freq="h"), "target": row}
                               for row in batch], freq="h")
        forecasts = list(predictor.predict(dataset))
        point = np.stack([f.mean for f in forecasts])
        quantiles = np.stack([np.stack([f.quantile(q) for q in [0.1, 0.5, 0.9]], -1) for f in forecasts])
        return point, quantiles
