"""Frozen TimesFM 2.5 adapter using the official timesfm API."""
import torch

from .foundation import FrozenFoundation


class TimesFM(FrozenFoundation):
    family = "timesfm"
    package = "timesfm"

    def load(self):
        import timesfm
        self.pipeline = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.path, torch_compile=False)
        self.pipeline.model.device = torch.device(self.device)
        self.pipeline.model.device_count = 1
        self.pipeline.model.to(self.device).eval().requires_grad_(False)
        self.compiled_for = None
        self.metadata["point_forecast"] = "native point head"
        self.metadata["forecast_flags"] = {"normalize_inputs": True, "continuous_quantile_head": True,
                                           "force_flip_invariance": True, "infer_is_positive": False,
                                           "fix_quantile_crossing": True, "torch_compile": False}

    def predict_batch(self, batch, horizon):
        import timesfm
        key = (batch.shape[-1], horizon)
        if self.compiled_for != key:
            self.pipeline.compile(timesfm.ForecastConfig(
                max_context=batch.shape[-1], max_horizon=horizon,
                per_core_batch_size=self.batch_size, normalize_inputs=True,
                use_continuous_quantile_head=True, force_flip_invariance=True,
                infer_is_positive=False, fix_quantile_crossing=True))
            self.compiled_for = key
        point, full = self.pipeline.forecast(horizon=horizon, inputs=list(batch))
        # Native output: point head, then q0.1 ... q0.9.
        return point, full[..., [1, 5, 9]]
