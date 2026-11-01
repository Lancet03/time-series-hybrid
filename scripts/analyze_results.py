"""Plot aggregate lead errors and one forecast window from an existing run."""
import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dataset", default="ETTm1")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--channel", default="0")
    parser.add_argument("--origin", type=int)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    metrics = pd.read_csv(run / "metrics.csv")
    metrics = metrics[(metrics.dataset == args.dataset) & (metrics.seed == args.seed)]
    selected = metrics[metrics.horizon == args.horizon]
    if selected.empty:
        parser.error("No results match the requested dataset, horizon and seed")
    first = run / f"{args.dataset}-{selected.iloc[0].model}-H{args.horizon}-s{args.seed}"
    with np.load(first / "origins.npz") as origins:
        origin = args.origin if args.origin is not None else int(origins["test"][0])
    label = f"{args.dataset}-H{args.horizon}-s{args.seed}-origin{origin}-channel{args.channel}"
    output = run / "analysis" / re.sub(r"[^A-Za-z0-9_-]", "_", label)
    output.mkdir(parents=True, exist_ok=True)

    horizons = sorted(metrics.horizon.unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(6 * len(horizons), 4), squeeze=False)
    for ax, horizon in zip(axes[0], horizons):
        for row in metrics[metrics.horizon == horizon].itertuples():
            task = run / f"{args.dataset}-{row.model}-H{horizon}-s{args.seed}"
            lead = pd.read_csv(task / "by_lead.csv")
            ax.plot(lead.lead, lead.mse, label=row.model)
        ax.set(title=f"{args.dataset}, H={horizon}, all test windows/channels",
               xlabel="Forecast step", ylabel="MSE (train-standardized)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "mse_by_lead.png", dpi=160)
    plt.close(fig)

    frames = {}
    for row in selected.itertuples():
        task = run / f"{args.dataset}-{row.model}-H{args.horizon}-s{args.seed}"
        # Predicate filtering avoids loading millions of forecast rows into pandas.
        frame = pd.read_parquet(task / "predictions.parquet", filters=[
            ("origin", "==", origin), ("channel", "==", args.channel),
        ]).sort_values("lead")
        if frame.empty:
            parser.error(f"No predictions for origin={origin}, channel={args.channel}: {task}")
        frames[row.model] = frame
    truth = next(iter(frames.values()))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(truth.lead, truth.y, color="black", linewidth=2, label="Observed")
    for name, frame in frames.items():
        ax.plot(frame.lead, frame.prediction, label=name)
    ax.set(title=f"{args.dataset}, channel={args.channel}, origin={origin}, H={args.horizon}",
           xlabel="Forecast step (15 minutes for ETTm1)", ylabel="Original units")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "forecast.png", dpi=160)
    plt.close(fig)

    for name, frame in frames.items():
        if "base_prediction" not in frame:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(frame.lead, frame.y, color="black", label="Observed")
        axes[0].plot(frame.lead, frame.base_prediction, label="ARIMA base")
        axes[0].plot(frame.lead, frame.prediction, label="Base + residual forecast")
        if {"q0.1", "q0.9"}.issubset(frame.columns):
            axes[0].fill_between(frame.lead, frame["q0.1"], frame["q0.9"], alpha=0.15,
                                 label="Hybrid q0.1-q0.9")
        axes[0].set(title=f"{name}: channel={args.channel}, origin={origin}", ylabel="Original units")
        axes[1].plot(frame.lead, frame.y - frame.base_prediction, label="Observed minus base")
        axes[1].plot(frame.lead, frame.correction, label="Chronos-2 correction")
        axes[1].axhline(0, color="gray", linewidth=0.7)
        axes[1].set(xlabel="Forecast step", ylabel="Correction (original units)")
        for ax in axes:
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output / f"{name}_components.png", dpi=160)
        plt.close(fig)
    print(f"Saved plots: {output}")
    print("The selected window illustrates behavior; use aggregate metrics to compare quality.")


if __name__ == "__main__":
    main()
