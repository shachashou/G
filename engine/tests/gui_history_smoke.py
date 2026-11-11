#!/usr/bin/env python3
"""Ticket 08: real desktop flow from candidate to v0 and default opponent."""
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / name) for name in ("gui", "tools", "trainer", "engine/tests")]
from PySide6.QtWidgets import QApplication
from value_fixture import build_pool
from value_train import TrainConfig, train_value
from model_registry import ModelRegistry
import gomoku_gui

EXE = str(Path(sys.argv[1]).resolve())


def pump_until(app, condition, timeout=40):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


def main():
    app = QApplication.instance() or QApplication([sys.argv[0]])
    with tempfile.TemporaryDirectory(prefix="gui-history-") as raw:
        root = Path(raw)
        pool = build_pool(root / "pool.sqlite3")
        for seed in (5, 6):
            summary = train_value(TrainConfig(pool=str(pool), output_dir=str(root / str(seed)),
                steps=8, batch_size=8, seed=seed, device="cpu"))
            assert summary["export"]["passed"]
        registry = root / "registry"
        window = gomoku_gui.MainWindow(EXE, 20, model=str(root / "5/model.gmvalue"),
            registry=registry, evaluations=root / "evaluations")
        window.show()
        panel = window.history_panel
        assert "尚未建立" in panel.current_label.text()
        panel.spin_ms.setValue(10)
        panel.btn_initial.click()
        assert window._phase == "evaluating"
        assert not panel.btn_initial.isEnabled()
        assert pump_until(app, lambda: window._preview is not None, 30), \
            "evaluation must publish a live board preview"
        window._new_game()
        assert "测评运行中" in window.error_label.text()
        assert pump_until(app, lambda: panel.thread is None)
        assert panel.last_result, panel.status.text()
        assert panel.last_result["publication"], panel.last_result.get("publication_error")
        assert "已登记 v0" in panel.status.text()
        assert "当前正式版本 v0" in window.eval_label.text()
        assert ModelRegistry(registry).read()["current"] == "v0"
        print("v0 registration, visible identity, busy protection: passed")
        # Small real history match must not be promoted merely for finishing.
        panel.edit_candidate.setText(str(root / "6/model.gmvalue"))
        panel.spin_pairs.setValue(1)
        panel.btn_challenge.click()
        assert pump_until(app, lambda: panel.thread is None)
        assert panel.last_result and not panel.last_result["publication"]
        assert "保留原正式版本" in panel.status.text()
        assert "insufficient_pairs" in panel.report.toPlainText(), panel.report.toPlainText()
        assert "严格胜率" in panel.report.toPlainText()
        assert "当前正式版本 v0" in window.eval_label.text()
        print("real historical match, insufficient evidence, incumbent retained: passed")
        panel.spin_pairs.setValue(64)
        panel.btn_challenge.click()
        panel.btn_stop.click()
        assert pump_until(app, lambda: panel.thread is None)
        assert panel.last_result and panel.last_result["status"] == "interrupted"
        assert ModelRegistry(registry).read()["current"] == "v0"
        print("stop and process cleanup: passed")

        # Pause an evaluation, then resume the same frozen run from the GUI.
        panel.spin_pairs.setValue(8)
        panel.edit_candidate.setText(str(root / "6/model.gmvalue"))
        panel.btn_challenge.click()
        assert pump_until(app, lambda: panel.thread is not None
                          and "局" in panel.status.text(), 15), panel.status.text()
        panel.btn_pause.click()
        assert pump_until(app, lambda: panel.thread is None, 60)
        assert panel.last_result["status"] == "interrupted", panel.last_result
        interrupted_run = panel.last_result["run"]
        assert panel.btn_resume_eval.isEnabled()
        panel.btn_resume_eval.click()
        assert pump_until(app, lambda: panel.thread is None, 120)
        resumed = panel.last_result
        assert resumed["run"] == interrupted_run, (resumed.get("run"), interrupted_run)
        assert resumed["status"] == "completed", resumed
        assert "保留原正式版本" in panel.status.text()
        assert ModelRegistry(registry).read()["current"] == "v0"
        print("pause evaluation, resume same frozen run, keep incumbent: passed")
        window.close()
        app.processEvents()
        # A fresh window follows the registry without an explicit --model.
        window = gomoku_gui.MainWindow(EXE, 20, registry=registry, evaluations=root / "evaluations")
        assert "当前正式版本 v0" in window.eval_label.text()
        assert window._about["value_model_loaded"]
        window.close()
        app.processEvents()
        print("default launch selects actual incumbent: passed")
    print("gui_history_smoke: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
