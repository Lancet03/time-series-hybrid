from html import escape
import json
from pathlib import Path

import pandas as pd


def compare_references(metrics, references):
    rows = []
    reference = pd.read_csv(references)
    for _, target in reference.iterrows():
        matches = metrics[(metrics.dataset == target.dataset) & (metrics.model == target.model)
                          & (metrics.horizon == target.horizon)] if not metrics.empty else metrics
        if matches.empty:
            rows.append({**target.to_dict(), "status": "not_run"})
            continue
        for _, actual in matches.iterrows():
            mismatch = []
            fields = ["protocol", "context", "seed", "channels", "steps", "dataset_sha256",
                      "train_end", "val_end", "epochs", "lr", "batch_size", "revin"]
            for key in fields:
                if key in target and pd.notna(target[key]):
                    value = actual.get(key)
                    equal = str(value) == str(target[key])
                    try:
                        equal = abs(float(value)-float(target[key])) < 1e-10
                    except (TypeError, ValueError):
                        pass
                    if not equal:
                        mismatch.append(key)
            if target.get("protocol") == "hn_mvts_code" and all(
                    key in target and pd.notna(target[key]) for key in ("steps", "val_end", "horizon")):
                expected_origins = int(target.steps)-int(target.val_end)-int(target.horizon)+1
                if actual.get("n_origins") != expected_origins:
                    mismatch.append("n_origins")
            comparable = not mismatch and not bool(actual.get("pilot", True))
            difference = float(actual.mse) - float(target.expected_mse)
            status = "protocol_mismatch" if not comparable else (
                "within_tolerance" if abs(difference) <= target.tolerance else "outside_tolerance")
            rows.append({**target.to_dict(), "run_id": actual.get("run_id"), "actual_mse": actual.mse,
                         "difference": difference, "status": status,
                         "mismatches": ", ".join(mismatch + (["pilot"] if actual.get("pilot", True) else []))})
    return pd.DataFrame(rows)


def build_report(run_dir, references=None):
    run_dir = Path(run_dir)
    path = run_dir / "metrics.csv"
    metrics = pd.read_csv(path) if path.exists() and path.stat().st_size > 1 else pd.DataFrame()
    failures = json.loads((run_dir / "failures.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    sections = ["<!doctype html><html lang='ru'><meta charset='utf-8'><title>Forecast experiment</title>",
                "<style>body{font:15px system-ui;margin:32px}table{border-collapse:collapse;font-size:13px}"
                "td,th{border:1px solid #ddd;padding:7px}th{background:#eef2f7}pre{white-space:pre-wrap}"
                "img{max-width:850px;width:100%}</style>",
                f"<h1>{escape(manifest['name'])}</h1>",
                f"<p>Run: {escape(run_dir.name)}. Status: {escape(manifest['status'])}. "
                f"Protocol: {escape(manifest['protocol'])}.</p>",
                "<p>MSE/MAE: train-standardized units. *_raw: original units. MASE: seasonal train scale. "
                "Coverage: nominal 80%. Hybrid intervals describe residual uncertainty conditional on the base point. "
                "Pilot/subset runs do not establish paper reproduction.</p>"]
    if not metrics.empty:
        fields = [c for c in ["dataset", "model", "horizon", "seed", "pilot", "n_origins", "mse", "mae",
                  "mase", "coverage_80", "width_80", "prediction_seconds", "selection_seconds",
                  "peak_rss_mb", "arima_nonconverged_fits"] if c in metrics]
        sections += ["<h2>Test metrics</h2>", metrics[fields].to_html(index=False, float_format=lambda v: f"{v:.6g}")]
        grouping = ["dataset", "model", "context", "horizon", "protocol", "pilot"]
        aggregate = metrics.groupby(grouping, dropna=False).agg(
            seeds=("seed", "nunique"), mse_mean=("mse", "mean"), mse_std=("mse", "std"),
            mae_mean=("mae", "mean"), mase_mean=("mase", "mean")).reset_index()
        aggregate.to_csv(run_dir / "aggregate.csv", index=False)
        sections += ["<h2>Seed aggregation (sample std)</h2>", aggregate.to_html(index=False)]
        if references:
            comparison = compare_references(metrics, references)
            comparison.to_csv(run_dir / "comparison.csv", index=False)
            fields = [c for c in ["dataset", "model", "horizon", "target_kind", "expected_mse", "actual_mse",
                                 "difference", "status", "mismatches"] if c in comparison]
            sections += ["<h2>Reference comparison</h2>", comparison[fields].to_html(index=False)]
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            for dataset, frame in aggregate.groupby("dataset"):
                fig, ax = plt.subplots(figsize=(8, 4))
                for model, group in frame.groupby("model"):
                    group = group.sort_values("horizon")
                    ax.plot(group.horizon, group.mse_mean, marker="o", label=model)
                ax.set(xlabel="Forecast horizon (steps)", ylabel="MSE (standardized)", title=dataset)
                ax.legend(fontsize=8)
                fig.tight_layout()
                filename = f"mse-{dataset}.png"
                fig.savefig(run_dir / filename, dpi=160)
                plt.close(fig)
                sections.append(f"<img src='{escape(filename)}' alt='MSE by horizon'>")
        except ImportError:
            sections.append("<p>Install matplotlib to render standalone charts.</p>")
    sections += ["<h2>Failures</h2>", f"<pre>{escape(json.dumps(failures, ensure_ascii=False, indent=2))}</pre>",
                 "<p><a href='metrics.csv'>metrics.csv</a> · <a href='manifest.json'>manifest.json</a> · "
                 "<a href='selection.json'>validation selection</a></p></html>"]
    (run_dir / "report.html").write_text("\n".join(sections), encoding="utf-8")
    return run_dir / "report.html"


def combine_runs(run_dirs, output):
    """Combine compatible model runs, preserving source IDs and refusing duplicate seeds."""
    import hashlib
    import numpy as np
    from .runner import write_json
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    frames, sources, failures = [], [], []
    signatures = {}
    for directory in map(lambda p: Path(p).resolve(), run_dirs):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "completed":
            raise ValueError(f"Only completed runs may be combined: {directory}")
        frame = pd.read_csv(directory / "metrics.csv")
        datasets = {d["name"]: d for d in manifest["datasets"]}
        for index, row in frame.iterrows():
            data = datasets[row.dataset]
            label = f"{row.dataset}-{row.model}-H{int(row.horizon)}-s{int(row.seed)}"
            with np.load(directory / label / "origins.npz") as positions:
                origin_hash = hashlib.sha256(positions["validation"].tobytes()+positions["test"].tobytes()).hexdigest()
            signature = (row.protocol, int(row.context), data["sha256"], tuple(data["channel_names"]),
                         data["train_end"], data["val_end"], data["test_end"], data["seasonality"], origin_hash)
            key = (row.dataset, int(row.horizon))
            if key in signatures and signature != signatures[key]:
                raise ValueError(f"Different data/protocol/origins for {key}: cannot compare these runs")
            signatures[key] = signature
            frame.loc[index, "origins_sha256"] = origin_hash
            frame.loc[index, "channels_signature"] = json.dumps(data["channel_names"])
        frames.append(frame)
        sources.append({"run_dir": str(directory), "run_id": manifest["run_id"],
                        "source_sha256": manifest["environment"]["source_sha256"]})
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["dataset", "model", "horizon", "seed"]).any():
        raise ValueError("Duplicate model/seed results: select one run explicitly")
    combined.to_csv(output / "metrics.csv", index=False)
    manifest = {"name": "Combined compatible experiments", "status": "completed",
                "protocol": combined.protocol.iloc[0], "sources": sources}
    write_json(output / "manifest.json", manifest)
    write_json(output / "selection.json", {"note": "See validation selection in each source run", "sources": sources})
    write_json(output / "failures.json", failures)
    return build_report(output)
