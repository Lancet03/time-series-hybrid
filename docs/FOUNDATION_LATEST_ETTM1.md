# TimesFM 3 and Chronos-2: ETTm1 zero-shot evaluation

Run commands from the `time-series-hybrid` directory in PowerShell.
These experiments freeze pretrained weights; they do not train or fine-tune either model.

The configuration preserves the previous common evaluation: context 336, horizons
48/96, all 7 channels, 57,600 time steps, 60/20/20 chronological split, stride 1.
Each channel is forecast independently; Chronos-2 cross-learning is disabled.
Both new adapters use q0.5 as the point forecast. TimesFM 3 uses symmetric
averaging, sorted quantiles and no positivity clamp on standardized data.
The old TimesFM 2.5 adapter uses its native point head, so point-forecast
definitions should be considered when interpreting differences across versions.

## Environment

```powershell
uv pip install --python .venv/Scripts/python.exe -e ".[chronos,timesfm,plots,network,dev]"

# If the environment has a CPU-only PyTorch build, restore the CUDA 13.0 build
# used in the previous experiment (requires a compatible NVIDIA driver).
uv pip install --python .venv/Scripts/python.exe --reinstall "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu130

uv pip check --python .venv/Scripts/python.exe
.\.venv\Scripts\python.exe -c "import torch, chronos, timesfm3; print(torch.__version__); assert torch.cuda.is_available(), 'CUDA unavailable'"
```

The required package versions are TimesFM 3.0.2 and chronos-forecasting 2.3.2.
Install using `pyproject.toml`; the historical `requirements-core.lock.txt`
describes the old environment and is not the lock for this experiment.

## Download and short check

```powershell
.\.venv\Scripts\python.exe -m tshybrid --system-certs prefetch configs/foundation_latest_ettm1.yaml
.\.venv\Scripts\python.exe -m tshybrid run configs/foundation_latest_ettm1_smoke.yaml
```

The smoke test uses two validation and two test windows per horizon. Its scores
are marked as pilot results and must not be compared with the full benchmark.

## Full evaluation

```powershell
.\.venv\Scripts\python.exe -m tshybrid run configs/foundation_latest_ettm1.yaml
```

To run a single model, append `--models chronos2` or `--models timesfm3`.
The runner evaluates both models sequentially in one process by default.
The full experiment has four tasks: two models times two horizons, seed 0.
Results are written to the new run directory printed at startup:
`metrics.csv`, `aggregate.csv`, `report.html`, manifests and forecasts.
MSE and MAE use train-standardized units; `mse_raw`/`mae_raw` use original units.

## ARIMA + Chronos-2 residual correction

The combined configuration compares four models: `arima`, `chronos2`,
`timesfm3`, and `arima_chronos2`, on the same evaluation windows.

```powershell
.\.venv\Scripts\python.exe -m tshybrid inspect configs/foundation_latest_hybrid_ettm1.yaml
.\.venv\Scripts\python.exe -u -m tshybrid run configs/foundation_latest_hybrid_ettm1_smoke.yaml
.\.venv\Scripts\python.exe -u -m tshybrid run configs/foundation_latest_hybrid_ettm1.yaml
```

To evaluate only ARIMA, Chronos-2 and their hybrid, append
`--models arima chronos2 arima_chronos2` to the full run command.
Selecting only `--models arima_chronos2` also runs its two dependencies.

For each channel and context, ARIMA fits its parameters on the first 128 of
336 points, then filters the remaining 208 points with fixed parameters.
Chronos-2 forecasts these 208 causal one-step residuals. The final forecast
is `ARIMA forecast + Chronos-2 residual forecast`. ARIMA's order is selected
from `(2,1,0)` and `(2,0,0)` using validation MSE, and the hybrid reuses
that selected base. Chronos-2 stays frozen.

This models ARIMA's errors; it does not explicitly decompose the series into
a smooth trend and a remainder. For that different hypothesis, a causal
trend extractor and its own evaluation protocol would be needed. See also
[ARIMA_CHRONOS.md](ARIMA_CHRONOS.md) for the residual construction.

The full configuration has eight tasks (four models times two horizons).
It uses the first 57,600 rows of the available ETTm1 file, with 34,560 train,
11,520 validation and 11,520 test rows. This is the project's common 60/20/20
protocol, rather than the standard ETT calendar split. Test counts are 11,473
windows for H=48 and 11,425 for H=96, each with all seven channels.
Repeated ARIMA fits make the full hybrid experiment substantially slower
than the foundation-only smoke test. Smoke scores verify execution only.

Hybrid `predictions.parquet` includes `base_prediction`, `correction` and
their scaled versions alongside the final prediction. Every run creates a
new output directory and prints its report path.

`configs/model-lock-foundation-latest.json` pins HF commits checked on 2026-10-31.
To deliberately refresh them later, run:

```powershell
.\.venv\Scripts\python.exe -m tshybrid --system-certs lock-models configs/foundation_latest_ettm1.yaml --output configs/model-lock-foundation-latest.json
```

References:
- https://github.com/google-research/timesfm
- https://github.com/amazon-science/chronos-forecasting

TimesFM 3 downloaded weights have the TimesFM non-commercial license;
see https://huggingface.co/google/timesfm-3.0-pytorch for its terms.
