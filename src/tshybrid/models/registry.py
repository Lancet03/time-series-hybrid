"""Supported standalone models; hybrids are composed by the experiment runner."""
from .ar import AR
from .arima import ARIMA
from .chronos import Chronos
from .dlinear import DLinear, DLinearModel
from .itransformer import InverseTransformer, ITransformerModel
from .moderntcn import ModernTCN, ModernTCNModel
from .moirai import Moirai
from .patchtst import PatchTST, PatchTSTModel
from .timesfm import TimesFM
from .tsmixer import TSMixer, TSMixerModel


NEURAL_MODELS = {
    "dlinear": (DLinear, {}, 50, 1e-4),
    "tsmixer": (TSMixer, {"d_hidden_temporal": 128, "d_hidden_channel": 24}, 25, 1e-4),
    "moderntcn": (ModernTCN, {"d_hidden": 256}, 15, 1e-4),
    "patchtst": (PatchTST, {"d_hidden": 32}, 10, 2e-4),
    "itransformer": (InverseTransformer, {"d_hidden": 256}, 25, 2e-4),
}

NEURAL_ADAPTERS = {
    "dlinear": DLinearModel,
    "tsmixer": TSMixerModel,
    "moderntcn": ModernTCNModel,
    "patchtst": PatchTSTModel,
    "itransformer": ITransformerModel,
}

STATISTICAL_MODELS = {"ar": AR, "arima": ARIMA}
FOUNDATION_MODELS = {"chronos": Chronos, "timesfm": TimesFM, "moirai": Moirai}
MODEL_TYPES = {*STATISTICAL_MODELS, *NEURAL_MODELS, *FOUNDATION_MODELS, "hybrid"}
