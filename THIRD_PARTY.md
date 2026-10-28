# Sources and attribution

This project implements DLinear, TSMixer, ModernTCN, PatchTST, iTransformer and
their shared layers locally in `src/tshybrid/models/`. The architectures and parts
of the implementation are adapted from
[HN-MVTS](https://github.com/av-savchenko/HN-MVTS), by Andrey Savchenko and Oleg Kachan,
under Apache-2.0, reference commit `e86c58a315576cef021d99e04b9b5fef55ddd6d6`.
The Apache-2.0 license is included in this project's `LICENSE`.

The local code is restructured into `networks.py`, `moderntcn.py` and `layers.py`;
training is handled by `neural.py`. It has no runtime imports from HN-MVTS or a
vendored copy. Reference comparisons in tests are optional and read the adjacent
checkout only when available. Hypernetwork variants are not implemented.

Layer names and parameter initialization order remain compatible with the
previously bundled Base networks. The separate `type: linear` is an additional
single-projection control. Lazy windows, training/checkpoint selection, streaming
evaluation and experiment provenance belong to this project.

Historical reports retain their actual implementation metadata: initial runs used
the earlier independent DLinear, and `reports/hn_models_laptop` used the bundled
HN-MVTS source. New runs record `implementation: time-series-hybrid`. Previous
reports and checkpoints are not rewritten. The earliest DLinear checkpoints use
different keys; checkpoints from the bundled Base networks remain compatible.

Architecture references: [DLinear](https://github.com/cure-lab/LTSF-Linear),
[RevIN](https://github.com/ts-kim/RevIN).

Foundation adapters call upstream packages, rather than copying model code:
[Chronos](https://github.com/amazon-science/chronos-forecasting),
[TimesFM](https://github.com/google-research/timesfm),
[Uni2TS / Moirai](https://github.com/SalesforceAIResearch/uni2ts).
Checkpoint licenses and dataset provenance remain attached to their upstream artifacts.
