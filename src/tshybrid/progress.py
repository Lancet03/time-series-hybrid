"""Throttled console progress and a small live JSON snapshot for each run."""
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter, sleep


def duration(seconds):
    if seconds is None:
        return "--"
    seconds = max(0, round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"



class Progress:
    def __init__(self, total, label, callback=None, interval=10.0, clock=perf_counter, **context):
        self.total, self.label = total, label
        self.callback, self.interval, self.clock = callback, interval, clock
        self.context = context
        self.started = self.last = clock()
        self.update(0, force=True)

    def update(self, completed, force=False):
        now = self.clock()
        if not force and completed < self.total and now - self.last < self.interval:
            return
        elapsed = now - self.started
        eta = elapsed * (self.total - completed) / completed if completed else None
        event = {**self.context, "stage": self.label, "completed": completed, "total": self.total,
                 "percent": 100 * completed / self.total if self.total else 100,
                 "elapsed_seconds": elapsed, "eta_seconds": eta}
        print(f"    {self.label}: {completed}/{self.total} ({event['percent']:.1f}%)"
              f" | elapsed={duration(elapsed)} | ETA={duration(eta)}", flush=True)
        if self.callback:
            self.callback(event)
        self.last = now


class RunProgress:
    def __init__(self, path, total_tasks, device, total_epochs=0):
        self.path = Path(path)
        self.started = perf_counter()
        self._warned = False
        self.state = {"status": "running", "device": device, "completed_tasks": 0,
                      "failed_tasks": 0, "total_tasks": total_tasks,
                      "completed_epochs": 0, "total_epochs": total_epochs}
        self.write()

    def write(self):
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.state["elapsed_seconds"] = perf_counter() - self.started
        temporary = self.path.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")
            for attempt in range(5):
                try:
                    temporary.replace(self.path)
                    return
                except PermissionError:
                    if attempt == 4:
                        raise
                    sleep(0.02)
        except OSError as error:
            # Telemetry is best-effort. A reader, antivirus or unavailable progress
            # path must never abort model fitting, evaluation or finalization.
            self.state["snapshot_write_errors"] = self.state.get("snapshot_write_errors", 0) + 1
            self.state["last_snapshot_error"] = str(error)
            if not self._warned:
                print("    Progress snapshot temporarily unavailable; computation continues. "
                      "Console and training history remain active.", flush=True)
                self._warned = True

    def begin(self, task, number, candidates):
        self.state.update(task=task, task_number=number, candidates=candidates, candidate=0,
                          activity={"stage": "initializing"})
        self.write()

    def candidate(self, number):
        self.state.update(candidate=number, activity={"stage": "initializing candidate"})
        self.write()

    def update(self, event):
        if event.get("stage") == "epoch completed":
            self.state["completed_epochs"] += 1
        self.state["activity"] = event
        self.write()

    def complete(self, success):
        self.state["completed_tasks"] += 1
        self.state["failed_tasks"] += int(not success)
        self.state["activity"] = {"stage": "completed" if success else "failed"}
        self.write()

    def finish(self, status):
        self.state["status"] = status
        self.write()
