import argparse
import json
from pathlib import Path
import shutil
import sys
from urllib.request import urlretrieve

import pandas as pd

from .data import file_hash, load_dataset, make_origins
from .reporting import build_report, combine_runs, compare_references
from .runner import read_config, run, validate_config, write_json


DATA_FILES = {"ETTm1": "ETTm1.npy", "ETTm2": "ETTm2.npy", "Weather": "weather.npy",
              "ECL": "electricity.npy"}


def prepare_data(source, destination, names):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    for name in names:
        if source:
            original = (Path(source) / DATA_FILES[name]).resolve()
            if not original.is_file():
                raise FileNotFoundError(original)
            target = destination / DATA_FILES[name]
            if target.exists() and file_hash(target) != file_hash(original):
                raise ValueError(f"Refusing to overwrite different data: {target}")
            if not target.exists():
                shutil.copyfile(original, target)
            manifest[name] = {"source": str(original), "file": target.name, "sha256": file_hash(target),
                              "provenance": "HN-MVTS author-provided NPY; preprocessing inherited, not independently verified"}
        else:
            if name not in {"ETTm1", "ETTm2"}:
                raise ValueError("Automatic official-source download supports ETTm1/ETTm2. Use --source for Weather/ECL")
            url = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv"
            target = destination / f"{name}.csv"
            if target.exists():
                raise ValueError(f"Already exists; inspect its provenance before replacing: {target}")
            temporary = target.with_suffix(".download")
            urlretrieve(url, temporary)
            # Check the schema before preserving a downloaded artifact.
            frame = pd.read_csv(temporary)
            if list(frame.columns) != ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]:
                raise ValueError("Unexpected ETT schema")
            temporary.replace(target)
            manifest[name] = {"source": url, "file": target.name, "sha256": file_hash(target)}
        print(f"{name}: {destination / manifest[name]['file']}")
    write_json(manifest_path, manifest)


def import_reference(repo, output):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    rows = []
    mapping = {"Weather": "weather.npy", "ECL": "electricity.npy"}
    for path in sorted((repo / "results").glob("2025*DLinear*.parquet")):
        frame = pd.read_parquet(path).reset_index()
        metrics = {c for c in frame if c.startswith(("mse_", "mae_"))}
        keys = [c for c in frame if c not in metrics | {"epoch_id"}]
        best = frame.loc[frame.groupby(keys, dropna=False).mse_val.idxmin()]
        for _, row in best[best.variant == "Base"].iterrows():
            dataset_path = repo / "data" / "datasets" / mapping.get(row.dataset, f"{row.dataset}.npy")
            if not dataset_path.exists():
                dataset_path = repo / "data" / dataset_path.name
            import numpy as np
            channels = np.load(dataset_path, mmap_mode="r").shape[1]
            fractions = [int(v)/10 for v in str(row.split)]
            size = int(row["size"])
            rows.append({"dataset": row.dataset, "model": "dlinear", "horizon": int(row.horizon),
                         "context": int(row.history), "protocol": "hn_mvts_code", "seed": int(row.seed),
                         "channels": channels, "steps": size, "train_end": int(size*fractions[0]),
                         "val_end": size-int(size*fractions[2]), "dataset_sha256": file_hash(dataset_path),
                         "epochs": int(row.epochs), "lr": row.lr, "batch_size": int(row.batch_size),
                         "revin": True, "expected_mse": row.mse_test, "tolerance": 0.005,
                         "target_kind": "author_log_single_seed", "source": path.name,
                         "source_sha256": file_hash(path), "validation_epoch": int(row.epoch_id)})
    if not rows:
        raise ValueError("No author DLinear logs found")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Imported {len(rows)} author reference rows: {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tsh")
    parser.add_argument("--system-certs", action="store_true", help="Use native OS certificate store (network extra)")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-data", help="Copy author NPY datasets or download official ETT CSV")
    prepare.add_argument("--source", type=Path)
    prepare.add_argument("--destination", type=Path, default=Path("data"))
    prepare.add_argument("--datasets", nargs="+", choices=DATA_FILES, default=["ETTm1", "Weather", "ECL"])
    inspect = sub.add_parser("inspect", help="Validate config and display data splits without loading models")
    inspect.add_argument("config", type=Path)
    execute = sub.add_parser("run")
    execute.add_argument("config", type=Path)
    execute.add_argument("--models", nargs="+")
    execute.add_argument("--datasets", nargs="+")
    report = sub.add_parser("report")
    report.add_argument("run_dir", type=Path)
    report.add_argument("--references", type=Path)
    hybrid_report = sub.add_parser("hybrid-report", help="Build a paired ARIMA / raw FM / residual FM study")
    hybrid_report.add_argument("run_dirs", nargs="+", type=Path)
    hybrid_report.add_argument("--output", type=Path, required=True)
    hybrid_report.add_argument("--reproduction", type=Path)
    compare = sub.add_parser("compare")
    compare.add_argument("run_dir", type=Path)
    compare.add_argument("--references", type=Path, required=True)
    combine = sub.add_parser("combine", help="Combine compatible runs from separate model environments")
    combine.add_argument("run_dirs", nargs="+", type=Path)
    combine.add_argument("--output", type=Path, required=True)
    reference = sub.add_parser("import-hn-reference")
    reference.add_argument("--repo", type=Path, default=Path("../HN-MVTS"))
    reference.add_argument("--output", type=Path, default=Path("references/hn_mvts_author.csv"))
    lock = sub.add_parser("lock-models")
    lock.add_argument("config", type=Path)
    lock.add_argument("--output", type=Path, default=Path("configs/model-lock.json"))
    prefetch = sub.add_parser("prefetch", help="Download locked checkpoints ahead of offline experiments")
    prefetch.add_argument("config", type=Path)
    prefetch.add_argument("--models", nargs="+")
    args = parser.parse_args(argv)
    if args.system_certs:
        import truststore
        truststore.inject_into_ssl()
    if args.command == "prepare-data":
        prepare_data(args.source, args.destination, args.datasets)
    elif args.command == "import-hn-reference":
        import_reference(args.repo, args.output)
    elif args.command == "inspect":
        config = read_config(args.config)
        validate_config(config)
        for spec in config["datasets"]:
            data = load_dataset(spec, args.config.resolve().parent)
            print(f"{data.name}: shape={data.values.shape}, train=[0,{data.train_end}), "
                  f"val=[{data.train_end},{data.val_end}), test=[{data.val_end},{len(data.values)})")
            for horizon in config["horizons"]:
                settings = config.get("evaluation", {})
                origins = make_origins(data.val_end, len(data.values), config["context"], horizon,
                                       settings.get("test_stride", 1), settings.get("test_limit"),
                                       config.get("protocol") == "hn_mvts_code")
                print(f"  H={horizon}: test origins={len(origins)}, first={origins[0]}, last={origins[-1]}")
    elif args.command == "run":
        _, success = run(args.config, args.models, args.datasets)
        if not success:
            sys.exit(1)
    elif args.command == "report":
        print(build_report(args.run_dir, args.references))
    elif args.command == "hybrid-report":
        from .study import study_report
        print(study_report(args.run_dirs, args.output, args.reproduction))
    elif args.command == "compare":
        result = compare_references(pd.read_csv(args.run_dir / "metrics.csv"), args.references)
        result.to_csv(args.run_dir / f"comparison-{args.references.stem}.csv", index=False)
        print(result.to_string(index=False))
    elif args.command == "combine":
        print(combine_runs(args.run_dirs, args.output))
    elif args.command == "lock-models":
        from huggingface_hub import HfApi
        config = read_config(args.config)
        entries = {}
        for spec in config["models"]:
            if spec["type"] in {"chronos", "timesfm", "moirai"}:
                params = spec["params"]
                info = HfApi().model_info(params["checkpoint"], revision=params.get("revision", "main"))
                entries[params["checkpoint"]] = {"revision": info.sha, "requested_revision": params.get("revision", "main")}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, entries)
        print(args.output)
    elif args.command == "prefetch":
        from .models.foundation import resolve_revision
        config = read_config(args.config)
        root = args.config.resolve().parent
        lock = json.loads((root / config["model_lock"]).read_text(encoding="utf-8"))
        for spec in config["models"]:
            if spec["type"] in {"chronos", "timesfm", "moirai"} and (
                    not args.models or spec["id"] in args.models):
                checkpoint = spec["params"]["checkpoint"]
                path, commit = resolve_revision(checkpoint, lock[checkpoint]["revision"],
                                                str((root / config.get("cache", "../.cache/huggingface")).resolve()))
                print(f"{spec['id']}: {commit} -> {path}")
