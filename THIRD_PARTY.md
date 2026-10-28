# Sources and attribution

DLinear, TSMixer, ModernTCN, PatchTST and iTransformer network code is copied from
[HN-MVTS](https://github.com/av-savchenko/HN-MVTS), by Andrey Savchenko and Oleg Kachan,
under Apache-2.0, reference commit `e86c58a315576cef021d99e04b9b5fef55ddd6d6`.
The Apache-2.0 license is included in this project's `LICENSE`.

Each architecture lives in its own module under `src/tshybrid/models/`:
`dlinear.py`, `tsmixer.py`, `moderntcn.py`, `patchtst.py`, `itransformer.py`.
Each module's header records the SHA256 of the copied source file.
Network definitions (including Hyper classes) retain their formulas, initialization
order and checkpoint keys. Import paths were made package-relative, trailing
whitespace was removed, and a common-evaluator adapter was appended to each file.
Required shared layers are copied from HN-MVTS `src/layers.py` into `layers.py`.
Base initialization, forward outputs and gradients were checked against the source
at transfer time. Current tests are standalone and do not load an external checkout.

`neural.py` supplies lazy training windows, Adam training and validation checkpoint
selection; the runner currently uses Base networks. Hyper classes are available
in their network modules, but Hyper/HyperE experiments are not exposed by the
common runner. AR/ARIMA, foundation API adapters, hybrid composition, streaming
evaluation and experiment provenance belong to this project.

LastValue, SeasonalNaive, ETS and the separate Linear control have been removed.
Historical reports and checkpoints retain their original implementation metadata;
new neural runs record `implementation: HN-MVTS` and the source path. The earliest
independent DLinear checkpoints use different keys; later Base checkpoints remain
compatible. Old reports have not been rewritten.

Foundation adapters call upstream packages rather than copying their network code:
[Chronos](https://github.com/amazon-science/chronos-forecasting),
[TimesFM](https://github.com/google-research/timesfm),
[Uni2TS / Moirai](https://github.com/SalesforceAIResearch/uni2ts).
Checkpoint licenses and dataset provenance remain attached to their upstream artifacts.
