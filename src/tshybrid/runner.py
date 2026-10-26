import copy
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import re
import subprocess
import threading
from time import perf_counter
import traceback
from uuid import uuid4
import zipfile

import numpy as np
import pandas as pd
import psutil
import torch
import yaml

from .data import load_dataset, make_origins
from .evaluation import evaluate
from .models.neural import DLinear, seed_everything
from .models.statistical import AR, ARIMA, ETS, LastValue, SeasonalNaive
from .reporting import build_report


def write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return None
        if isinstance(item, np.generic):
            return item.item()
        return item
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def merge(left, right):
    result = copy.deepcopy(left)
    for key, value in right.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def read_config(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError("Cyclic config inheritance")
    seen.add(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping")
    if "extends" in config:
        parent = path.parent / config.pop("extends")
        # All data/output/cache paths resolve relative to the leaf config directory.
        config = merge(read_config(parent, seen), config)
    return config


def validate_config(config):
    if config.get("protocol", "common") not in {"common", "hn_mvts_code"}:
        raise ValueError("Unknown evaluation protocol")
    if config.get("purpose") not in {"pilot", "benchmark", "reproduction"}:
        raise ValueError("purpose must be pilot, benchmark or reproduction")
    if not config.get("datasets") or not config.get("models"):
        raise ValueError("At least one dataset and model are required")
    if int(config["context"]) < 1 or not config.get("horizons") or min(config["horizons"]) < 1:
        raise ValueError("Positive context/horizons are required")
    for entries in [config["datasets"], config["models"]]:
        identifiers = [item.get("id", item.get("name")) for item in entries]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Duplicate dataset/model IDs")
        if any(not isinstance(i, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", i) for i in identifiers):
            raise ValueError("Dataset/model IDs must contain only letters, numbers, underscores and hyphens")
    known = {}
    for spec in config["models"]:
        if config.get("protocol") == "hn_mvts_code" and spec["type"] == "dlinear":
            params = spec.get("params", {})
            if (params.get("kernel", 25) != 25 or not params.get("revin", True)
                    or config.get("evaluation", {}).get("batch_size", 16) != params.get("batch_size", 64)):
                raise ValueError("HN code protocol requires kernel=25, RevIN and matching train/evaluation batch sizes")
        if spec["type"] == "hybrid":
            for key in ("base", "residual"):
                if spec[key] not in known:
                    raise ValueError(f"Hybrid {key} must refer to a preceding model")
            if known[spec["residual"]]["type"] not in {"chronos", "timesfm", "moirai"}:
                raise ValueError("Hybrid residual must be a frozen foundation model")
            if known[spec["base"]]["type"] not in {"last_value", "seasonal_naive", "ar", "arima"}:
                raise ValueError("Hybrid base must implement causal residuals: last_value, seasonal_naive, AR or ARIMA")
            if any(len(known[spec[key]].get("seeds", [0])) != 1 for key in ("base", "residual")):
                raise ValueError("Hybrid dependencies must specify one unambiguous seed")
        if not spec.get("seeds", [0]):
            raise ValueError("Seeds cannot be empty")
        known[spec["id"]] = spec


class PeakMemory:
    """Sample whole-process RSS during a model task, including already cached models."""
    def __enter__(self):
        self.stop = threading.Event()
        self.peak = 0
        self.process = psutil.Process()
        def sample():
            while not self.stop.is_set():
                self.peak = max(self.peak, self.process.memory_info().rss)
                self.stop.wait(0.05)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


def environment():
    source_root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        digest.update(str(path.relative_to(source_root)).encode())
        digest.update(path.read_bytes())
    repo = source_root.parent.parent
    try:
        git = subprocess.run(["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo),
                              "rev-parse", "HEAD"], capture_output=True, text=True)
        commit = git.stdout.strip() if git.returncode == 0 else None
    except OSError:
        commit = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "packages": {d.metadata["Name"]: d.version for d in distributions()},
            "git_commit": commit,
            "source_sha256": digest.hexdigest()}


def make_model(spec, params, data, context, horizon, seed, device, cache_dir, cache, selected):
    kind = spec["type"]
    if kind == "last_value":
        return LastValue()
    if kind == "seasonal_naive":
        return SeasonalNaive(params.get("period", data.seasonality))
    if kind == "ar":
        return AR(**params)
    if kind == "arima":
        return ARIMA(**params)
    if kind == "ets":
        return ETS(**params)
    if kind in {"dlinear", "linear"}:
        return DLinear(context, horizon, len(data.channels), seed, device, linear=kind == "linear", **params)
    if kind in {"chronos", "timesfm", "moirai"}:
        from .models.foundation import Foundation
        key = json.dumps([kind, params, device], sort_keys=True)
        if key not in cache:
            cache[key] = Foundation(kind, device=device, cache_dir=cache_dir, **params)
        return cache[key]
    if kind == "hybrid":
        from .models.hybrid import Hybrid
        missing = [spec[key] for key in ("base", "residual") if spec[key] not in selected]
        if missing:
            raise ValueError(f"Hybrid dependencies failed or were not run: {missing}")
        return Hybrid(selected[spec["base"]], selected[spec["residual"]], **params)
    raise ValueError(f"Unknown model type {kind}")


def run(config_path, model_ids=None, dataset_names=None):
    config_path = Path(config_path).resolve()
    config = read_config(config_path)
    if dataset_names:
        unknown = set(dataset_names) - {d["name"] for d in config["datasets"]}
        if unknown:
            raise ValueError(f"Unknown datasets: {unknown}")
        config["datasets"] = [d for d in config["datasets"] if d["name"] in dataset_names]
    if model_ids:
        wanted = set(model_ids)
        specs = {s["id"]: s for s in config["models"]}
        if wanted - specs.keys():
            raise ValueError(f"Unknown models: {wanted - specs.keys()}")
        for spec in reversed(config["models"]):
            if spec["id"] in wanted and spec["type"] == "hybrid":
                wanted.update([spec["base"], spec["residual"]])
        config["models"] = [s for s in config["models"] if s["id"] in wanted]
    validate_config(config)
    root = config_path.parent
    if config.get("model_lock"):
        lock = json.loads((root / config["model_lock"]).read_text(encoding="utf-8"))
        for spec in config["models"]:
            if spec["type"] in {"chronos", "timesfm", "moirai"}:
                entry = lock[spec["params"]["checkpoint"]]
                spec["params"]["revision"] = entry["revision"]
    context, protocol = config["context"], config.get("protocol", "common")
    evaluation = config.get("evaluation", {})
    legacy = protocol == "hn_mvts_code"
    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(config.get("threads", 4))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_dir = (root / config.get("output", "../outputs") / run_id).resolve()
    run_dir.mkdir(parents=True)
    with zipfile.ZipFile(run_dir / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(Path(__file__).parent.rglob("*.py")):
            archive.write(path, "tshybrid/" + path.relative_to(Path(__file__).parent).as_posix())
    manifest = {"name": config["name"], "run_id": run_id, "status": "running", "protocol": protocol,
                "config": config, "config_path": str(config_path), "environment": environment(),
                "datasets": [], "models": [], "memory_note": "whole-process sampled RSS, includes cached FMs"}
    write_json(run_dir / "manifest.json", manifest)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Run directory: {run_dir}", flush=True)
    all_metrics, selections, failures, cache = [], [], [], {}
    try:
        for dataset_spec in config["datasets"]:
            data = load_dataset(dataset_spec, root)
            manifest["datasets"].append(data.manifest)
            for horizon in config["horizons"]:
                val_origins = make_origins(data.train_end, data.val_end, context, horizon,
                                          evaluation.get("val_stride", 1), evaluation.get("val_limit"), legacy)
                test_origins = make_origins(data.val_end, len(data.values), context, horizon,
                                           evaluation.get("test_stride", 1), evaluation.get("test_limit"), legacy)
                selected = {}
                for spec in config["models"]:
                    for seed in spec.get("seeds", [0]):
                        task = {"dataset": data.name, "model": spec["id"], "horizon": horizon, "seed": seed}
                        label = f"{data.name}-{spec['id']}-H{horizon}-s{seed}"
                        task_dir = run_dir / label
                        task_dir.mkdir()
                        print(f"  {label}", flush=True)
                        try:
                            with PeakMemory() as memory:
                                seed_everything(seed)
                                start = perf_counter()
                                best, best_score, best_params, best_timing = None, float("inf"), None, {}
                                for candidate, overrides in enumerate(spec.get("candidates", [{}])):
                                    try:
                                        params = {**spec.get("params", {}), **overrides}
                                        seed_everything(seed)
                                        initialization_start = perf_counter()
                                        model = make_model(spec, params, data, context, horizon, seed, device,
                                                           str((root / config.get("cache", "../.cache/huggingface")).resolve()),
                                                           cache, selected)
                                        initialization_seconds = perf_counter() - initialization_start
                                        if spec["type"] in {"chronos", "timesfm", "moirai"}:
                                            # Initialization and cache hits must not change the sampling RNG.
                                            seed_everything(seed)
                                        fit_start = perf_counter()
                                        model.fit(data, val_origins, protocol)
                                        fit_seconds = perf_counter() - fit_start
                                        validation_start = perf_counter()
                                        val_metrics, _, _ = evaluate(model, data, val_origins, context, horizon,
                                                                     evaluation.get("batch_size", 16), legacy)
                                        timing = {"model_init_seconds": initialization_seconds,
                                                  "fit_seconds": fit_seconds,
                                                  "validation_seconds": perf_counter() - validation_start}
                                        selections.append({**task, "candidate": candidate, "params": params,
                                                           "validation": val_metrics, **timing})
                                        if hasattr(model, "history"):
                                            pd.DataFrame(model.history).to_csv(task_dir / f"training-{candidate}.csv", index=False)
                                        if val_metrics["mse"] < best_score:
                                            best, best_score, best_params = model, val_metrics["mse"], params
                                            best_timing = timing
                                    except np.linalg.LinAlgError as error:
                                        selections.append({**task, "candidate": candidate,
                                                           "params": params, "status": "numerical_failure",
                                                           "error": str(error)})
                                        print(f"    candidate {candidate}: numerical failure: {error}", flush=True)
                                if best is None:
                                    raise ValueError("No candidate has finite validation MSE")
                                selection_seconds = perf_counter() - start
                                if spec["id"] not in selected:
                                    selected[spec["id"]] = best
                                seed_everything(seed + 1000)
                                scores, leads, predict_seconds = evaluate(
                                    best, data, test_origins, context, horizon,
                                    evaluation.get("batch_size", 16), legacy, task_dir / "predictions.parquet")
                                leads.to_csv(task_dir / "by_lead.csv", index=False)
                                np.savez(task_dir / "origins.npz", validation=val_origins, test=test_origins)
                                if hasattr(best, "net"):
                                    torch.save(best.net.state_dict(), task_dir / "best.pt")
                            pilot = (config["purpose"] == "pilot" or bool(evaluation.get("val_limit"))
                                     or bool(evaluation.get("test_limit")) or bool(dataset_spec.get("channels"))
                                     or bool(best_params.get("train_limit"))
                                     or best_params.get("train_stride", 1) != 1
                                     or (legacy and (evaluation.get("val_stride", 1) != 1
                                                     or evaluation.get("test_stride", 1) != 1)))
                            record = {**task, **scores, "run_id": run_id, "context": context,
                                      "protocol": protocol, "pilot": pilot, "channels": len(data.channels),
                                      "steps": len(data.values), "train_end": data.train_end, "val_end": data.val_end,
                                      "dataset_sha256": data.manifest["sha256"],
                                      "channels_signature": json.dumps(data.channels),
                                      "origins_sha256": hashlib.sha256(val_origins.tobytes()+test_origins.tobytes()).hexdigest(),
                                      "seasonality": data.seasonality,
                                      "n_origins": len(test_origins), "selection_seconds": selection_seconds,
                                      "prediction_seconds": predict_seconds, "peak_rss_mb": memory.peak/1024**2,
                                      **best_timing,
                                      **{k: best_params[k] for k in ("epochs", "lr", "batch_size", "revin") if k in best_params},
                                      "arima_nonconverged_fits": getattr(best, "nonconverged_fits", 0)}
                            all_metrics.append(record)
                            model_manifest = {**task, "selected_params": best_params, "metadata": best.metadata,
                                              "validation_mse": best_score}
                            manifest["models"].append(model_manifest)
                            write_json(task_dir / "model.json", model_manifest)
                            print(f"    test MSE={scores['mse']:.6f} MAE={scores['mae']:.6f} MASE={scores['mase']:.6f}", flush=True)
                        except Exception as error:
                            failures.append({**task, "error": str(error), "traceback": traceback.format_exc()})
                            print(f"    FAILED: {error}", flush=True)
                        pd.DataFrame(all_metrics).to_csv(run_dir / "metrics.csv", index=False)
                        write_json(run_dir / "selection.json", selections)
                        write_json(run_dir / "failures.json", failures)
                        write_json(run_dir / "manifest.json", manifest)
                        gc.collect()
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        failures.append({"error": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        if manifest["status"] == "running":
            manifest["status"] = "completed_with_failures" if failures else "completed"
        write_json(run_dir / "failures.json", failures)
        write_json(run_dir / "selection.json", selections)
        write_json(run_dir / "manifest.json", manifest)
        references = root / config["references"] if config.get("references") else None
        build_report(run_dir, references)
        print(f"Report: {run_dir / 'report.html'}", flush=True)
    return run_dir, not failures
