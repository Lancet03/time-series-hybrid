"""Prepare independent model terminals and combine their explicitly recorded results."""
import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback


MODELS = ("ar", "arima", "dlinear", "tsmixer", "moderntcn", "patchtst",
          "itransformer", "chronos", "timesfm", "moirai")


def save(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.02)


def prepare(config_path, output, dataset="ETTm1", root=None):
    import copy
    import yaml
    from .runner import read_config, validate_config

    root = Path(root or Path.cwd()).resolve()
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = read_config(config_path)
    validate_config(config)
    datasets = [d for d in config["datasets"] if d["name"] == dataset]
    if not datasets:
        raise ValueError(f"Dataset not in config: {dataset}")
    specs = {s["id"]: s for s in config["models"]}
    if set(MODELS) - specs.keys():
        raise ValueError("The parallel plan requires all ten standalone models")
    executable = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    interpreters = {m: root / (".venv-moirai" if m == "moirai" else ".venv") / executable for m in MODELS}
    for path in set(interpreters.values()):
        if not path.is_file():
            raise FileNotFoundError(f"Environment not found: {path}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "configs").mkdir()
    (output / "state").mkdir()
    (output / "logs").mkdir()
    if config.get("model_lock"):
        shutil.copy2(config_path.parent / config["model_lock"], output / "model-lock.json")
    entries = []
    for name in MODELS:
        cfg = copy.deepcopy(config)
        cfg["datasets"] = copy.deepcopy(datasets)
        for spec in cfg["datasets"]:
            spec["path"] = str((config_path.parent / spec["path"]).resolve())
        cfg["models"] = [copy.deepcopy(specs[name])]
        cfg["output"] = str(output / "runs" / name)
        cfg["cache"] = str((config_path.parent / config.get("cache", "../.cache/huggingface")).resolve())
        if config.get("model_lock"):
            cfg["model_lock"] = str(output / "model-lock.json")
        # Ten simultaneous jobs share one machine. GPU work stays on CUDA/auto;
        # CPU models use CPU explicitly, with bounded internal thread counts.
        cfg["device"] = "cpu" if name in {"ar", "arima", "moirai"} else config.get("device", "auto")
        cfg["threads"] = 2 if name == "moirai" else 1
        if name == "arima":
            cfg["models"][0].setdefault("params", {}).setdefault("workers", 4)
        validate_config(cfg)
        path = output / "configs" / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        entries.append({"model": name, "python": str(interpreters[name]), "config": str(path),
                        "run_root": cfg["output"], "device": cfg["device"], "threads": cfg["threads"]})
        save(output / "state" / f"{name}.json", {"model": name, "status": "pending"})
        (output / f"{name}.ps1").write_text(
            f"$plan = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'plan.json') -Raw | ConvertFrom-Json\n"
            f"$entry = $plan.entries | Where-Object {{ $_.model -eq '{name}' }}\n"
            "$ErrorActionPreference = 'Continue'\n"
            "try { $Host.UI.RawUI.WindowTitle = $entry.model + ' | ' + $plan.dataset } catch {}\n"
            "Set-Location -LiteralPath $plan.root\n"
            f"& $entry.python -u -m tshybrid.parallel run-model --batch $PSScriptRoot --model '{name}'\n"
            "if ($LASTEXITCODE -ne 0) { Write-Warning 'Model failed; inspect this terminal and the batch log.' }\n",
            encoding="utf-8")
    plan = {"root": str(root), "dataset": dataset, "created_at": datetime.now(timezone.utc).isoformat(),
            "entries": entries}
    save(output / "plan.json", plan)
    (output / "Start-All.ps1").write_text('''$ErrorActionPreference = 'Stop'
$plan = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'plan.json') -Raw | ConvertFrom-Json
$shell = (Get-Process -Id $PID).Path
foreach ($entry in $plan.entries) {
    $state = Get-Content -LiteralPath (Join-Path $PSScriptRoot ('state/' + $entry.model + '.json')) -Raw | ConvertFrom-Json
    if ($state.status -eq 'completed') { Write-Host ($entry.model + ': already completed'); continue }
    if ($state.status -eq 'running' -and (Get-Process -Id $state.pid -ErrorAction SilentlyContinue)) {
        Write-Host ($entry.model + ': already running'); continue
    }
    $script = Join-Path $PSScriptRoot ($entry.model + '.ps1')
    $arguments = @('-NoProfile', '-NoExit', '-ExecutionPolicy', 'Bypass', '-File', ('"' + $script + '"'))
    Start-Process -FilePath $shell -ArgumentList $arguments -WorkingDirectory $plan.root -WindowStyle Normal
}
Write-Host 'All ten terminals opened. Use Status.ps1 to inspect progress, Combine.ps1 -Wait to wait and combine.'
''', encoding="utf-8")
    (output / "Status.ps1").write_text('''$plan = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'plan.json') -Raw | ConvertFrom-Json
Set-Location -LiteralPath $plan.root
& $plan.entries[0].python -m tshybrid.parallel status --batch $PSScriptRoot
''', encoding="utf-8")
    (output / "Combine.ps1").write_text('''param([switch]$Wait)
$plan = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'plan.json') -Raw | ConvertFrom-Json
Set-Location -LiteralPath $plan.root
$arguments = @('-m', 'tshybrid.parallel', 'combine', '--batch', $PSScriptRoot)
if ($Wait) { $arguments += '--wait' }
& $plan.entries[0].python @arguments
''', encoding="utf-8")
    return output


def read_plan(batch):
    return json.loads((Path(batch) / "plan.json").read_text(encoding="utf-8"))


@contextmanager
def exclusive(path):
    with Path(path).open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.terminal, name)


def run_model(batch, model):
    batch = Path(batch).resolve()
    plan = read_plan(batch)
    entry = next(e for e in plan["entries"] if e["model"] == model)
    with exclusive(batch / "state" / f"{model}.lock"):
        state = {"model": model, "status": "running", "pid": os.getpid(),
                 "started_at": datetime.now(timezone.utc).isoformat()}
        save(batch / "state" / f"{model}.json", state)
        os.chdir(plan["root"])
        # Limit native CPU pools before importing torch/numpy in this process.
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[variable] = str(entry["threads"])
        if entry["device"] == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        with (batch / "logs" / f"{model}.log").open("a", encoding="utf-8") as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                print(f"=== {model} started {state['started_at']} | device={entry['device']} ===", flush=True)
                try:
                    from .runner import run
                    directory, success = run(Path(entry["config"]))
                    state.update(run_dir=str(directory), status="completed" if success else "failed",
                                 exit_code=0 if success else 1)
                except KeyboardInterrupt:
                    state.update(status="interrupted", exit_code=130)
                except Exception as error:
                    traceback.print_exc()
                    state.update(status="failed", error=str(error), exit_code=1)
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        save(batch / "state" / f"{model}.json", state)
        return state["exit_code"]


def states(batch):
    import psutil
    batch = Path(batch)
    rows = []
    for entry in read_plan(batch)["entries"]:
        row = json.loads((batch / "state" / f"{entry['model']}.json").read_text(encoding="utf-8"))
        if row["status"] == "running" and not psutil.pid_exists(row["pid"]):
            row.update(status="failed", error="Worker process exited without final status")
        candidates = ([Path(row["run_dir"]) / "progress.json"] if row.get("run_dir") else
                      sorted(Path(entry["run_root"]).glob("*/progress.json"), key=lambda p: p.stat().st_mtime))
        if candidates:
            try:
                row["progress"] = json.loads(candidates[-1].read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        rows.append(row)
    return rows


def combine(batch, wait=False):
    from .reporting import combine_runs
    batch = Path(batch).resolve()
    while True:
        rows = states(batch)
        failed = [r["model"] for r in rows if r["status"] in {"failed", "interrupted"}]
        if failed:
            raise ValueError(f"Rerun failed/interrupted models before combining: {', '.join(failed)}")
        pending = [r["model"] for r in rows if r["status"] != "completed"]
        if not pending:
            return combine_runs([Path(r["run_dir"]) for r in rows], batch / "combined")
        if not wait:
            raise ValueError(f"Models still pending/running: {', '.join(pending)}")
        print(f"Waiting for {len(pending)} models: {', '.join(pending)}", flush=True)
        time.sleep(10)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("prepare")
    create.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    create.add_argument("--dataset", default="ETTm1")
    create.add_argument("--output", type=Path, required=True)
    for name in ("run-model", "status", "combine"):
        command = sub.add_parser(name)
        command.add_argument("--batch", type=Path, required=True)
        if name == "run-model":
            command.add_argument("--model", choices=MODELS, required=True)
        if name == "combine":
            command.add_argument("--wait", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            print(prepare(args.config, args.output, args.dataset))
        elif args.command == "run-model":
            return run_model(args.batch, args.model)
        elif args.command == "combine":
            print(combine(args.batch, args.wait))
        else:
            for row in states(args.batch):
                progress = row.get("progress", {})
                activity = progress.get("activity", {})
                print(f"{row['model']:14} {row['status']:12} "
                      f"task={progress.get('task_number', '-')}/{progress.get('total_tasks', '-')} "
                      f"epochs={progress.get('completed_epochs', '-')}/{progress.get('total_epochs', '-')} "
                      f"{activity.get('stage', '')}")
    except (ValueError, OSError, StopIteration) as error:
        parser.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
