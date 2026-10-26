"""Run points 5/6 serially in the two prepared environments, then build the report."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def execute(python, arguments, log_path):
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    process = subprocess.Popen([str(python), "-m", "tshybrid", *arguments], cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", env=environment)
    run_dir = None
    with log_path.open("w", encoding="utf-8") as log:
        for line in process.stdout:
            log.write(line)
            log.flush()
            if line.startswith("Run directory: "):
                run_dir = Path(line.removeprefix("Run directory: ").strip())
            if (line.startswith(("Run directory:", "Report:", "  ", "C:\\workspace"))
                    and "site-packages" not in line):
                print(line.rstrip(), flush=True)
    if process.wait() != 0:
        raise RuntimeError(f"Experiment failed; full log: {log_path}")
    return run_dir


def completed(path):
    path = path.resolve()
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise ValueError(f"Cannot resume from an incomplete run: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "hybrid_study")
    parser.add_argument("--resume-core", type=Path)
    parser.add_argument("--resume-moirai", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Report directory already exists; specify a new --output: {args.output}")
    executable = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    core_python, moirai_python = ROOT / ".venv" / executable, ROOT / ".venv-moirai" / executable
    for path in (core_python, moirai_python):
        if not path.is_file():
            parser.error(f"Create the documented environment first: {path}")
    log_dir = ROOT / "outputs" / ("study-logs-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    log_dir.mkdir(parents=True)
    core = completed(args.resume_core) if args.resume_core else execute(
        core_python, ["run", "configs/hybrid_core.yaml"], log_dir / "core.log")
    moirai = completed(args.resume_moirai) if args.resume_moirai else execute(
        moirai_python, ["run", "configs/hybrid_moirai.yaml"], log_dir / "moirai.log")
    command = ["hybrid-report", str(core), str(moirai), "--output", str(args.output.resolve())]
    for name in ("reproduction_corrected", "reproduction"):
        reproduction = ROOT / "reports" / name
        if reproduction.is_dir():
            command += ["--reproduction", str(reproduction)]
            break
    execute(core_python, command, log_dir / "report.log")
    print(f"Finished: {args.output.resolve() / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
