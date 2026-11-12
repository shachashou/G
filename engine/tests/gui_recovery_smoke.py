#!/usr/bin/env python3
"""Ticket 12 offscreen GUI acceptance: unified tabs, pause, restart and resume."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ("gui", "tools", "trainer")]
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
import gomoku_gui
import rapfi_teacher
from model_registry import ModelRegistry
from sample_pool import SamplePool

EXE = str(Path(sys.argv[1]).resolve()) if len(sys.argv) > 1 else str(
    ROOT / "build/engine/gomoku_nnue.exe")


def wait(app, predicate, seconds=120):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def configure(panel, root):
    panel.btn_quick.click()
    panel.edit_output.setText(str(root / "tasks"))
    panel.spin_min_train.setValue(1)
    panel.spin_new.setValue(1)
    panel.spin_steps.setValue(4000)
    panel.spin_batch.setValue(4)
    panel.spin_rounds.setValue(1)
    panel.spin_games.setValue(10)
    panel.spin_pairs.setValue(1)
    panel.spin_ms.setValue(10)
    panel.spin_rounds.setValue(1)
    panel.combo_device.setCurrentText("cpu")
    config = replace(
        panel.make_config(), validation_games=1, max_pending_samples=2,
        min_round_interval_s=0, collection_delay_ms=10, seed=1,
        max_minutes=2,
        collection={"depth": 2, "max_plies": 24, "opening_plies": 2,
                    "opening_region": 7, "sample_every": 3, "budget_ms": 100,
                    "vcf_nodes": 1000, "vcf_plies": 4},
        evaluation={"pairs": 1, "movetime_ms": 10, "vcf_policy": "off"})
    panel.make_config = lambda: config
    return config


def main():
    if not rapfi_teacher.check_install(rapfi_teacher.load_lock()).ok:
        print("SKIP pinned Rapfi is not installed")
        return 77
    app = QApplication.instance() or QApplication([sys.argv[0]])
    with tempfile.TemporaryDirectory(prefix="gui-recovery-") as raw:
        root = Path(raw)
        window = gomoku_gui.MainWindow(EXE, 20, pool=root / "pool.sqlite3",
                                       registry=root / "registry",
                                       evaluations=root / "evaluations")
        window.show()
        labels = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        assert labels == ["对弈", "训练", "测评"], labels
        window.tabs.setCurrentIndex(1)
        assert window.training_mode.current() in ("collect", "fromzero", "selfplay")
        panel = window.training_task_panel
        config = configure(panel, root)
        panel.btn_start.click()
        assert wait(app, lambda: panel.task is not None
                   and panel.task.snapshot().get("trainer", {}).get(
                       "progress", {}).get("step", 0) >= 2, 120), panel.status.text()
        panel.btn_pause.click()
        assert wait(app, lambda: panel.task.snapshot()["status"] == "paused", 90), \
            panel.status.text()
        assert panel.btn_resume.isEnabled() and not panel.btn_pause.isEnabled()
        paused_round = json.loads(
            (Path(panel.task.snapshot()["directory"]) / "round-00000.json"
             ).read_text("utf-8"))
        assert paused_round["status"] == "paused", paused_round
        assert (Path(panel.task.snapshot()["directory"])
                / "candidates/round-00000/last.pt").is_file()
        panel.btn_resume.click()
        assert wait(app, lambda: panel.thread is None, 180), panel.status.text()
        first = panel.last_result
        assert first and first["status"] == "completed", first
        assert first["rounds"][0]["status"] == "published"
        assert ModelRegistry(config.registry).resolve()["version"] == "v0"
        print("GUI pause at optimizer checkpoint -> resume -> v0: passed")

        # Start a second task, pause it, then close the client. Closing must
        # reap the worker and leave a resumable state.
        panel.make_config = lambda: replace(config, max_rounds=2)
        panel.btn_start.click()
        assert wait(app, lambda: panel.task is not None
                   and panel.task.snapshot().get("trainer", {}).get(
                       "progress", {}).get("step", 0) >= 2, 120), panel.status.text()
        second_task = panel.task
        second_directory = Path(second_task.snapshot()["directory"])
        panel.btn_pause.click()
        assert wait(app, lambda: second_task.snapshot()["status"] == "paused", 90)
        window.close()
        app.processEvents()
        assert panel.thread is None or not panel.thread.is_alive()
        stopped = second_task.snapshot()
        assert stopped["status"] == "stopped", stopped
        assert stopped["cleanup_complete"]
        with SamplePool(config.pool, create=False) as pool:
            assert pool.stats()["staged_games"] == 0 and not pool.validate()
        assert not Path(str(config.pool) + ".training-task.lock").exists()
        print("close reaps workers, keeps paused checkpoint, releases lease: passed")

        reopened = gomoku_gui.MainWindow(EXE, 20, pool=root / "pool.sqlite3",
                                         registry=root / "registry",
                                         evaluations=root / "evaluations")
        reopened.show()
        panel = reopened.training_task_panel
        panel.edit_output.setText(str(root / "tasks"))
        panel._refresh_recoverable()
        index = panel.combo_recover.findData(str(second_directory))
        assert index >= 0, panel.combo_recover.count()
        panel.combo_recover.setCurrentIndex(index)
        panel.btn_recover.click()
        assert wait(app, lambda: panel.thread is None, 180), panel.status.text()
        resumed = panel.last_result
        assert resumed and resumed["status"] == "completed", resumed
        assert resumed["cleanup_complete"]
        assert len(ModelRegistry(config.registry).read()["versions"]) == 1
        print("restart -> discover task -> resume checkpoint -> retained v0: passed")
        reopened.close()
        app.processEvents()
    print("gui_recovery_smoke: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
