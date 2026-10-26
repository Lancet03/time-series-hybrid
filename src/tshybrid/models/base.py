from dataclasses import dataclass, field

import numpy as np


@dataclass
class Forecast:
    # All shapes: [windows, channels, horizon]. Quantiles are optional.
    point: np.ndarray
    quantiles: dict = field(default_factory=dict)
    components: dict = field(default_factory=dict)

    def validate(self, expected):
        if self.point.shape != expected or not np.isfinite(self.point).all():
            raise ValueError(f"Invalid forecast shape/values: {self.point.shape}, expected {expected}")
        for level, values in self.quantiles.items():
            if not 0 < level < 1 or values.shape != expected or not np.isfinite(values).all():
                raise ValueError(f"Invalid quantile forecast {level}")
        for name, values in self.components.items():
            if values.shape != expected or not np.isfinite(values).all():
                raise ValueError(f"Invalid forecast component {name}")
        return self


class Model:
    metadata = {}

    def fit(self, *args, **kwargs):
        return self
