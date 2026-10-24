# Hybrid Time Series Forecasting

Research project focused on hybrid time series forecasting using statistical models and foundation models for residual prediction.

## Research Goal

To investigate whether zero-shot foundation models can improve statistical forecasting methods by predicting their residuals.

## Planned Experiments

- **Statistical models:** AR, ARIMA, ETS, Seasonal Naive
- **Neural baselines:** DLinear, PatchTST
- **Foundation models:** Chronos, TimesFM, Moirai
- **Hybrid approaches:** Statistical forecasting + foundation model residual correction
- **Datasets:** ETT, Electricity, Exchange Rate, M4
- **Evaluation metrics:** MSE, MAE, MASE

## Research Objectives

1. Reproduce baseline forecasting results.
2. Implement hybrid forecasting architectures.
3. Analyze when residual correction improves forecasting accuracy.
4. Develop a criterion for selecting when to apply residual correction.

## Status

Work in progress. Initial stage: baseline implementation and benchmark reproduction.
