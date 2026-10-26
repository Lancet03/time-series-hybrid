"""Frozen, channel-independent official FM APIs; immutable HF revisions are recorded."""
import numpy as np
import torch

from .base import Forecast, Model


def resolve_revision(checkpoint, revision, cache_dir):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    options = dict(repo_id=checkpoint, revision=revision, cache_dir=cache_dir,
                   allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model"])
    try:
        path = snapshot_download(**options, local_files_only=True)
    except LocalEntryNotFoundError:
        path = snapshot_download(**options)
    # snapshot_download resolves even a moving branch to a concrete commit directory.
    from pathlib import Path
    return path, Path(path).name


class Foundation(Model):
    def __init__(self, family, checkpoint, revision, device="cpu", cache_dir=None,
                 batch_size=16, num_samples=100, patch_size=32):
        self.family, self.device = family, device
        # Check optional dependencies before downloading a potentially large checkpoint.
        from importlib.util import find_spec
        package = {"chronos": "chronos", "timesfm": "timesfm", "moirai": "uni2ts"}.get(family)
        if package is None or find_spec(package) is None:
            raise ImportError(f"Install the {family!r} project extra in this environment first")
        self.batch_size, self.num_samples, self.patch_size = batch_size, num_samples, patch_size
        self.path, commit = resolve_revision(checkpoint, revision, cache_dir)
        self.metadata = {"family": family, "checkpoint": checkpoint, "requested_revision": revision,
                         "resolved_revision": commit, "device": device, "frozen": True,
                         "channel_mode": "independent", "quantiles": [0.1, 0.5, 0.9]}
        if family == "chronos":
            from chronos import BaseChronosPipeline
            self.pipeline = BaseChronosPipeline.from_pretrained(self.path, device_map=device,
                                                               torch_dtype=torch.float32)
            self.pipeline.model.eval().requires_grad_(False)
            self.metadata["point_forecast"] = "median (Chronos-Bolt API mean is the median)"
            self.metadata["native_horizon"] = self.pipeline.model.chronos_config.prediction_length
            self.metadata["long_horizon_policy"] = "allow upstream autoregressive quantile rollout"
        elif family == "timesfm":
            import timesfm
            self.pipeline = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.path, torch_compile=False)
            self.pipeline.model.device = torch.device(device)
            self.pipeline.model.device_count = 1
            self.pipeline.model.to(device).eval().requires_grad_(False)
            self.compiled_for = None
            self.metadata["point_forecast"] = "native point head"
            self.metadata["forecast_flags"] = {"normalize_inputs": True, "continuous_quantile_head": True,
                                               "force_flip_invariance": True, "infer_is_positive": False,
                                               "fix_quantile_crossing": True, "torch_compile": False}
        elif family == "moirai":
            from uni2ts.model.moirai import MoiraiModule
            self.module = MoiraiModule.from_pretrained(self.path).to(device).eval().requires_grad_(False)
            self.metadata.update(point_forecast="sample mean", num_samples=num_samples, patch_size=patch_size)
        else:
            raise ValueError(f"Unknown foundation family: {family}")

    def predict(self, x, horizon):
        shape = (*x.shape[:-1], horizon)
        flat = x.reshape(-1, x.shape[-1])
        points, quantile_batches = [], []
        for start in range(0, len(flat), self.batch_size):
            batch = flat[start:start+self.batch_size]
            with torch.inference_mode():
                if self.family == "chronos":
                    quantiles, point = self.pipeline.predict_quantiles(
                        torch.tensor(batch, dtype=torch.float32), prediction_length=horizon,
                        quantile_levels=[0.1, 0.5, 0.9], limit_prediction_length=False)
                    point, quantiles = point.cpu().numpy(), quantiles.cpu().numpy()
                elif self.family == "timesfm":
                    import timesfm
                    key = (x.shape[-1], horizon)
                    if self.compiled_for != key:
                        self.pipeline.compile(timesfm.ForecastConfig(
                            max_context=x.shape[-1], max_horizon=horizon,
                            per_core_batch_size=self.batch_size, normalize_inputs=True,
                            use_continuous_quantile_head=True, force_flip_invariance=True,
                            infer_is_positive=False, fix_quantile_crossing=True))
                        self.compiled_for = key
                    point, full = self.pipeline.forecast(horizon=horizon, inputs=list(batch))
                    # Native output: point head, then q0.1 ... q0.9.
                    quantiles = full[..., [1, 5, 9]]
                else:
                    import pandas as pd
                    from gluonts.dataset.common import ListDataset
                    from uni2ts.model.moirai import MoiraiForecast
                    model = MoiraiForecast(module=self.module, prediction_length=horizon,
                                           context_length=x.shape[-1], patch_size=self.patch_size,
                                           num_samples=self.num_samples, target_dim=1,
                                           feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
                    predictor = model.create_predictor(batch_size=self.batch_size, device=self.device)
                    dataset = ListDataset([{"start": pd.Period("2000-01-01", freq="h"), "target": row}
                                           for row in batch], freq="h")
                    forecasts = list(predictor.predict(dataset))
                    point = np.stack([f.mean for f in forecasts])
                    quantiles = np.stack([np.stack([f.quantile(q) for q in [0.1, 0.5, 0.9]], -1)
                                          for f in forecasts])
            points.append(np.asarray(point)[..., :horizon])
            quantile_batches.append(np.asarray(quantiles)[:, :horizon])
        point = np.concatenate(points).reshape(shape)
        quantiles = np.concatenate(quantile_batches).reshape(*shape, 3)
        return Forecast(point, {q: quantiles[..., i] for i, q in enumerate([0.1, 0.5, 0.9])})
