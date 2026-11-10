#!/usr/bin/env python3
"""Offscreen GUI smoke test for value-network training (ticket 06).

Builds a small learnable pool, starts a real training run from the window,
checks the progress/result labels and the exported model, then starts a long
run and stops it, checking that the checkpoint/export path still completes and
the UI returns to idle.

Usage: python gui_train_smoke.py <gomoku_nnue.exe>
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "gui"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HERE))

from PySide6.QtWidgets import QApplication  # noqa: E402

import gomoku_gui  # noqa: E402
from value_fixture import build_pool  # noqa: E402

EXE = sys.argv[1] if len(sys.argv) > 1 else str(
    ROOT / "build" / "engine" / "gomoku_nnue.exe")
failures = 0


def check(cond, msg):
    global failures
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures += 1


def pump(app, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)


def wait_for(app, predicate, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.03)
    return False


def main():
    tmp = Path(tempfile.mkdtemp(prefix="gui-train-"))
    pool = build_pool(tmp / "pool.sqlite3")
    app = QApplication.instance() or QApplication([sys.argv[0]])
    window = gomoku_gui.MainWindow(EXE, 100)
    window.show()
    pump(app, 0.3)

    window.edit_pool.setText(str(pool))
    window.edit_train_output.setText(str(tmp / "run"))
    window.spin_train_steps.setValue(12)
    window.spin_train_batch.setValue(8)
    window.spin_train_lr.setValue(0.002)
    window._start_training()
    finished = wait_for(app, lambda: "导出保真度" in window.train_result.text()
                        or "训练失败" in window.error_label.text(), 180)
    check(finished, "training run finished and reported a result")
    check(window._phase == "idle", "GUI returned to idle after training")
    check("val MAE" in window.train_result.text(),
          "result panel shows validation MAE")
    check("通过" in window.train_result.text(),
          "result panel shows the export fidelity result")
    model_path = tmp / "run" / "model.gmvalue"
    report_path = tmp / "run" / "fidelity.json"
    check(model_path.is_file(), "GUI training exported the model")
    check(report_path.is_file(), "GUI training wrote the fidelity report")
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        check(report["fidelity"]["export_pass"] is True,
              "export fidelity passed for the GUI run")

    print("[stop training]")
    window.edit_train_output.setText(str(tmp / "stopped"))
    window.spin_train_steps.setValue(20000)
    window._start_training()
    started = wait_for(app, lambda: "step" in window.train_label.text(), 30)
    check(started, "long training reported progress")
    window._stop_training()
    stopped = wait_for(app, lambda: window._phase == "idle", 120)
    check(stopped, "stop returns the GUI to idle")
    check("已停止" in window.train_result.text()
          or "完成" in window.train_result.text(),
          "stopped training reports its final state")
    check((tmp / "stopped" / "last.pt").is_file(),
          "stopping saved a checkpoint")
    check((tmp / "stopped" / "model.gmvalue").is_file(),
          "stopping still exported the model")

    window.close()
    pump(app, 0.1)
    print()
    if failures:
        print(f"{failures} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
