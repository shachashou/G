#!/usr/bin/env python3
"""Real one-click teacher task in an isolated offscreen desktop window."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ("gui", "tools", "trainer")]
from PySide6.QtWidgets import QApplication
import gomoku_gui
import rapfi_teacher
from sample_pool import SamplePool
from model_registry import ModelRegistry


def wait(app, predicate, seconds=90):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(.01)
    return False


def main():
    if not rapfi_teacher.check_install(rapfi_teacher.load_lock()).ok:
        print("SKIP pinned Rapfi is not installed")
        return 77
    app = QApplication.instance() or QApplication([sys.argv[0]])
    with tempfile.TemporaryDirectory(prefix="gui-teacher-task-") as raw:
        root = Path(raw)
        window = gomoku_gui.MainWindow(str(Path(sys.argv[1]).resolve()), 20,
            pool=root / "pool.sqlite3", registry=root / "registry", evaluations=root / "evaluations")
        window.show()
        panel = window.training_task_panel
        panel.btn_quick.click()
        panel.edit_output.setText(str(root / "tasks"))
        panel.spin_min_train.setValue(1)
        panel.spin_new.setValue(1)
        panel.spin_steps.setValue(4)
        panel.spin_batch.setValue(4)
        panel.spin_rounds.setValue(2)
        panel.spin_games.setValue(30)
        panel.spin_pairs.setValue(1)
        panel.spin_ms.setValue(10)
        panel.combo_device.setCurrentText("cpu")
        config = replace(panel.make_config(), validation_games=1, max_pending_samples=2,
            min_round_interval_s=0, collection_delay_ms=20, seed=1, max_minutes=2,
            collection={"depth":2,"max_plies":24,"opening_plies":2,"opening_region":7,
                "sample_every":3,"budget_ms":100,"vcf_nodes":5000,"vcf_plies":4},
            evaluation={"pairs":1,"movetime_ms":10,"vcf_policy":"off"})
        panel.make_config = lambda: config
        panel.btn_start.click()
        assert window._phase == "continuous"
        assert not panel.btn_start.isEnabled()
        assert not panel.spin_steps.isEnabled()
        assert wait(app, lambda: window._preview is not None, 60), \
            "background collection must publish a live board preview"
        assert len(window._preview["cells"]) == 225
        window._new_game()
        assert "持续训练运行中" in window.error_label.text()
        window.history_panel.start(False)
        assert "请先停止" in window.history_panel.status.text()
        assert wait(app, lambda: panel.thread is None), panel.status.text()
        result = panel.last_result
        assert result and result["status"] == "completed", result
        assert len(result["rounds"]) == 2
        assert result["rounds"][0]["evaluation"]["publication"]["version"] == "v0"
        assert not result["rounds"][1]["evaluation"]["publication"]
        assert "当前正式版本 v0" in window.eval_label.text()
        assert "池容量" in panel.metrics.toPlainText()
        assert "insufficient_pairs" in panel.metrics.toPlainText()
        assert panel.btn_start.isEnabled() and not panel.btn_stop.isEnabled()
        with SamplePool(config.pool, create=False) as pool:
            assert pool.stats()["staged_games"] == 0 and not pool.validate()
        print("one click -> collection/training -> v0 -> challenge rejection -> actual play model: passed")
        original = ModelRegistry(config.registry).resolve()["sha256"]
        panel.make_config = lambda: replace(config, max_rounds=10)
        panel.btn_start.click()
        panel.btn_stop.click()
        assert wait(app, lambda: panel.thread is None)
        assert panel.last_result["status"] == "stopped", panel.last_result
        assert panel.last_result["cleanup_complete"]
        assert ModelRegistry(config.registry).resolve()["sha256"] == original
        print("stop retains committed data and incumbent, worker cleanup: passed")
        window.close()
        app.processEvents()
    print("gui_training_task_smoke: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
