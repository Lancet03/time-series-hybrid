# Sources and attribution

The DLinear decomposition and affine RevIN equations in `models/neural.py` follow
[HN-MVTS](https://github.com/av-savchenko/HN-MVTS), by Andrey Savchenko and Oleg Kachan,
distributed under Apache-2.0. The original license is available in this project's
`LICENSE` (Apache-2.0). This implementation adds lazy windows, configurable training,
streaming evaluation and explicit experiment provenance. It does not implement HN hypernetworks.

Architecture references: [DLinear](https://github.com/cure-lab/LTSF-Linear),
[RevIN](https://github.com/ts-kim/RevIN).

Foundation adapters call upstream packages, rather than copying model code:
[Chronos](https://github.com/amazon-science/chronos-forecasting),
[TimesFM](https://github.com/google-research/timesfm),
[Uni2TS / Moirai](https://github.com/SalesforceAIResearch/uni2ts).
Checkpoint licenses and dataset provenance remain attached to their upstream artifacts.
