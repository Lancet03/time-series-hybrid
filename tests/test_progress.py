"""Live progress must be accurate, bounded and available before training completes."""
import json

import numpy as np
import pandas as pd
import pytest

from tshybrid.data import load_dataset, make_origins
from tshybrid.evaluation import evaluate
from tshybrid.models.dlinear import DLinearModel
from tshybrid.progress import Progress, RunProgress
from helpers import Persistence


def test_progress_throttles_updates_and_reports_stage_eta():
    now = [0.0]
    events = []
    progress = Progress(10, "test", events.append, interval=10, clock=lambda: now[0])
    now[0] = 5
    progress.update(3)
    assert len(events) == 1
    now[0] = 12
    progress.update(5)
    assert events[-1]["percent"] == 50
    assert events[-1]["eta_seconds"] == pytest.approx(12)
    now[0] = 16
    progress.update(10)
    assert events[-1]["eta_seconds"] == 0
    assert events[-1]["percent"] == 100


def test_run_snapshot_tracks_failed_tasks_and_interruption(tmp_path):
    path = tmp_path / "progress.json"
    reporter = RunProgress(path, 2, "cpu")
    reporter.begin({"model": "arima", "horizon": 48, "seed": 0}, 1, 2)
    reporter.candidate(1)
    reporter.update({"stage": "validation", "completed": 10, "total": 100})
    assert json.loads(path.read_text())["activity"]["completed"] == 10
    reporter.complete(False)
    reporter.begin({"model": "dlinear", "horizon": 48, "seed": 0}, 2, 1)
    reporter.finish("interrupted")
    state = json.loads(path.read_text())
    assert state["status"] == "interrupted"
    assert state["completed_tasks"] == state["failed_tasks"] == 1
    assert state["task"]["model"] == "dlinear"


def test_training_history_is_readable_after_each_epoch(tmp_path):
    np.save(tmp_path / "series.npy", np.random.default_rng(7).normal(size=(120, 2)))
    data = load_dataset({"name": "test", "path": "series.npy", "seasonality": 2,
                         "split": [.6, .2, .2]}, tmp_path)
    origins = make_origins(data.train_end, data.val_end, 12, 4, limit=3)
    model = DLinearModel(12, 4, 2, epochs=2, batch_size=4, train_limit=8)
    model.history_path = tmp_path / "training-0.csv"
    seen = []

    def inspect(event):
        if event["stage"] == "epoch completed":
            history = pd.read_csv(model.history_path)
            assert len(history) == event["epoch"]
            assert history.iloc[-1].eta_seconds >= 0
            seen.append(event["epoch"])

    model.progress_callback = inspect
    model.fit(data, origins, "common")
    assert seen == [1, 2]
    events = []
    expected, _, _ = evaluate(Persistence(), data, origins, 12, 4, batch_size=2)
    actual, _, _ = evaluate(Persistence(), data, origins, 12, 4, batch_size=2,
                            progress_label="test windows", progress_callback=events.append)
    assert actual == expected
    assert events[-1]["completed"] == len(origins)
    assert events[-1]["percent"] == 100


def test_open_progress_file_cannot_abort_updates(tmp_path):
    reporter = RunProgress(tmp_path / "progress.json", 1, "cpu")
    # On Windows a regular open reader prevents replacing the file. Progress
    # must keep its in-memory state and retry later, without aborting the run.
    with reporter.path.open(encoding="utf-8"):
        reporter.begin({"model": "dlinear"}, 1, 1)
        reporter.finish("completed")
    reporter.write()
    state = json.loads(reporter.path.read_text())
    assert state["status"] == "completed"
    assert state["task"]["model"] == "dlinear"


def test_progress_retries_a_transient_windows_lock(tmp_path, monkeypatch):
    from pathlib import Path
    import tshybrid.progress as module
    reporter = RunProgress(tmp_path / "progress.json", 1, "cpu")
    replace = Path.replace
    attempts = []

    def transient(path, target):
        attempts.append(1)
        if len(attempts) <= 2:
            raise PermissionError("Simulated WinError 5")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", transient)
    monkeypatch.setattr(module, "sleep", lambda seconds: None)
    reporter.begin({"model": "dlinear"}, 1, 1)
    assert len(attempts) == 3
    assert json.loads(reporter.path.read_text())["task"]["model"] == "dlinear"


def test_blocked_progress_file_cannot_abort_training(tmp_path, monkeypatch, capsys):
    from pathlib import Path
    import tshybrid.progress as module
    np.save(tmp_path / "series.npy", np.random.default_rng(7).normal(size=(120, 2)))
    data = load_dataset({"name": "test", "path": "series.npy", "seasonality": 2,
                         "split": [.6, .2, .2]}, tmp_path)
    origins = make_origins(data.train_end, data.val_end, 12, 4, limit=3)
    reporter = RunProgress(tmp_path / "progress.json", 1, "cpu", total_epochs=2)
    replace = Path.replace

    def blocked(path, target):
        raise PermissionError("Simulated persistent WinError 5")

    monkeypatch.setattr(Path, "replace", blocked)
    monkeypatch.setattr(module, "sleep", lambda seconds: None)
    model = DLinearModel(12, 4, 2, epochs=2, batch_size=4, train_limit=8)
    model.progress_callback = reporter.update
    model.history_path = tmp_path / "training.csv"
    reporter.begin({"model": "dlinear"}, 1, 1)
    model.fit(data, origins, "common")
    reporter.complete(True)
    reporter.finish("completed")
    assert reporter.state["completed_epochs"] == 2
    assert reporter.state["snapshot_write_errors"] > 0
    assert len(pd.read_csv(model.history_path)) == 2
    assert capsys.readouterr().out.count("Progress snapshot temporarily unavailable") == 1
    # The next successful snapshot includes all work done while the file was locked.
    monkeypatch.setattr(Path, "replace", replace)
    reporter.write()
    state = json.loads(reporter.path.read_text())
    assert state["status"] == "completed"
    assert state["completed_epochs"] == 2
