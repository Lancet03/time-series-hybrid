from .base import Forecast, Model


class Hybrid(Model):
    def __init__(self, base, residual, warmup=128):
        if not isinstance(warmup, int) or warmup < 1:
            raise ValueError("Residual warmup must be a positive integer")
        self.base, self.residual, self.warmup = base, residual, warmup
        self.nonconverged_fits = 0
        self.metadata = {"warmup": warmup, "base": base.metadata, "residual": residual.metadata,
                         "interval_interpretation": "FM residual uncertainty conditional on base point forecast",
                         "history_policy": "residuals built entirely inside the same original context",
                         "point_forecast": "base + residual FM point"}

    def predict(self, x, horizon):
        if self.warmup >= x.shape[-1]:
            raise ValueError("Residual warmup must be shorter than context")
        previous = getattr(self.base, "nonconverged_fits", 0)
        base, residuals = self.base.components(x, horizon, self.warmup)
        self.nonconverged_fits += getattr(self.base, "nonconverged_fits", 0) - previous
        correction = self.residual.predict(residuals.astype("float32"), horizon)
        return Forecast(base.point + correction.point,
                        {q: base.point + value for q, value in correction.quantiles.items()},
                        {"base": base.point, "correction": correction.point})
