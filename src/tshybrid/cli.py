import argparse
import json
from pathlib import Path
import shutil
import sys
from urllib.request import urlretrieve

import pandas as pd

from .data import file_hash, load_dataset, make_origins
from .reporting import build_report, combine_runs
from .models.registry import FOUNDATION_MODELS
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
                              "provenance": "User-provided NPY; preprocessing supplied by the dataset provider"}
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


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tsh")
    parser.add_argument("--system-certs", action="store_true", help="Use native OS certificate store (network extra)")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-data", help="Copy local NPY datasets or download official ETT CSV")
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
    combine = sub.add_parser("combine", help="Combine compatible runs from separate model environments")
    combine.add_argument("run_dirs", nargs="+", type=Path)
    combine.add_argument("--output", type=Path, required=True)
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
        print(build_report(args.run_dir))
    elif args.command == "combine":
        print(combine_runs(args.run_dirs, args.output))
    elif args.command == "lock-models":
        from huggingface_hub import HfApi
        config = read_config(args.config)
        entries = {}
        for spec in config["models"]:
            if spec["type"] in FOUNDATION_MODELS:
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
            if spec["type"] in FOUNDATION_MODELS and (
                    not args.models or spec["id"] in args.models):
                checkpoint = spec["params"]["checkpoint"]
                path, commit = resolve_revision(checkpoint, lock[checkpoint]["revision"],
                                                str((root / config.get("cache", "../.cache/huggingface")).resolve()))
                print(f"{spec['id']}: {commit} -> {path}")
