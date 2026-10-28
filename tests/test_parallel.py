"""Preparing parallel terminals must not start training or select unrelated runs."""
import json
import os
from pathlib import Path
import shutil

import pytest

from tshybrid.parallel import MODELS, combine, exclusive, prepare, read_plan, run_model, save
from tshybrid.runner import read_config


@pytest.fixture
def batch(tmp_path):
    root = tmp_path / "project with spaces"
    directory = root / "configs"
    directory.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "configs"
    for name in ("benchmark.yaml", "model-lock.json"):
        shutil.copy2(source / name, directory / name)
    executable = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    for name in (".venv", ".venv-moirai"):
        path = root / name / executable
        path.parent.mkdir(parents=True)
        path.touch()
    return prepare(directory / "benchmark.yaml", root / "outputs" / "batch", root=root)


def test_prepare_preserves_full_protocol_and_creates_ten_pending_models(batch):
    plan = read_plan(batch)
    assert [entry["model"] for entry in plan["entries"]] == list(MODELS)
    original = read_config(Path(plan["root"]) / "configs" / "benchmark.yaml")
    for entry in plan["entries"]:
        cfg = read_config(Path(entry["config"]))
        assert cfg["horizons"] == original["horizons"]
        assert cfg["evaluation"] == original["evaluation"]
        assert [d["name"] for d in cfg["datasets"]] == ["ETTm1"]
        reference = next(s for s in original["models"] if s["id"] == entry["model"])
        assert cfg["models"] == [reference]
        assert Path(cfg["datasets"][0]["path"]).is_absolute()
        assert Path(cfg["output"]) == batch / "runs" / entry["model"]
        assert (batch / (entry["model"] + ".ps1")).exists()
        assert json.loads((batch / "state" / (entry["model"] + ".json")).read_text())["status"] == "pending"
        assert (".venv-moirai" in entry["python"]) == (entry["model"] == "moirai")
    assert not (batch / "runs").exists()
    with pytest.raises(FileExistsError):
        prepare(Path(plan["root"]) / "configs/benchmark.yaml", batch, root=plan["root"])


def test_worker_records_exact_output_and_keeps_a_log(batch, monkeypatch):
    import tshybrid.runner as runner
    plan = read_plan(batch)
    monkeypatch.chdir(plan["root"])
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(key, os.environ.get(key, "1"))
    directory = batch / "runs" / "dlinear" / "specific-result"

    def fake_run(config):
        assert Path(config).name == "dlinear.yaml"
        print("epoch 1/50")
        directory.mkdir(parents=True, exist_ok=True)
        return directory, True

    monkeypatch.setattr(runner, "run", fake_run)
    assert run_model(batch, "dlinear") == 0
    state = json.loads((batch / "state/dlinear.json").read_text())
    assert state["status"] == "completed"
    assert state["run_dir"] == str(directory)
    assert "epoch 1/50" in (batch / "logs/dlinear.log").read_text()

    def fail_run(config):
        raise RuntimeError("Simulated CUDA out of memory")

    monkeypatch.setattr(runner, "run", fail_run)
    assert run_model(batch, "dlinear") == 1
    assert json.loads((batch / "state/dlinear.json").read_text())["status"] == "failed"
    monkeypatch.setattr(runner, "run", fake_run)
    assert run_model(batch, "dlinear") == 0
    assert json.loads((batch / "state/dlinear.json").read_text())["status"] == "completed"
    assert "Simulated CUDA out of memory" in (batch / "logs/dlinear.log").read_text()


def test_combining_requires_all_models_and_uses_recorded_directories(batch, monkeypatch):
    import tshybrid.reporting as reporting
    with pytest.raises(ValueError, match="pending/running"):
        combine(batch)
    expected = []
    for name in MODELS:
        directory = batch / "runs" / name / "chosen-result"
        expected.append(directory)
        save(batch / "state" / f"{name}.json", {"model": name, "status": "completed", "run_dir": str(directory)})

    def fake_combine(directories, output):
        assert directories == expected
        assert output == batch / "combined"
        return output / "report.html"

    monkeypatch.setattr(reporting, "combine_runs", fake_combine)
    assert combine(batch) == batch / "combined/report.html"
    save(batch / "state/moirai.json", {"model": "moirai", "status": "failed"})
    with pytest.raises(ValueError, match="Rerun failed/interrupted"):
        combine(batch)


def test_model_lock_prevents_duplicates_and_releases_after_exit(tmp_path):
    path = tmp_path / "model.lock"
    with exclusive(path):
        with pytest.raises(OSError):
            with exclusive(path):
                pytest.fail("Duplicate model lock was acquired")
    with exclusive(path):
        pass
