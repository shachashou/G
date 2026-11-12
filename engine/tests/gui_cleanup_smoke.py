#!/usr/bin/env python3
"""Offscreen acceptance for the cleanup dialog: models and historical data."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ("gui", "tools", "trainer")]
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication, QMessageBox
import gomoku_gui
from cleanup_dialog import CleanupDialog

EXE = str(Path(sys.argv[1]).resolve()) if len(sys.argv) > 1 else str(
    ROOT / "build/engine/gomoku_nnue.exe")


def pump(app, seconds=0.1):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)


def bundle(path: Path, size: int = 64, stamp: float = 0.0) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.gmvalue").write_bytes(b"m" * size)
    (path / "last.pt").write_bytes(b"p" * size)
    if stamp:
        os.utime(path / "model.gmvalue", (stamp, stamp))
    return path


def task(path: Path, status: str, rounds: int = 0, stamp: float = 0.0) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "state.json").write_text(json.dumps({"status": status}),
                                     encoding="utf-8")
    (path / "games").mkdir(exist_ok=True)
    (path / "games" / "game-00001.json").write_bytes(b"g" * 64)
    for index in range(rounds):
        bundle(path / "candidates" / f"round-{index:05d}", stamp=stamp + index)
    if stamp:
        os.utime(path, (stamp, stamp))
    return path


def run(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "report.json").write_text("{}", encoding="utf-8")
    return path


def build(base: Path) -> dict:
    weights = base / "weights"
    evaluations = base / "data" / "evaluations"
    registry = weights / "registry"
    registry.mkdir(parents=True)
    guard = base / "guard-registry"
    guard.mkdir()
    referenced = run(evaluations / "run-ref")
    stale = run(evaluations / "run-stale")
    (guard / "state.json").write_text(json.dumps({
        "format": "GMVALUE-REGISTRY1", "current": "v1",
        "versions": [{"version": "v1", "model": "versions/v1/model.gmvalue",
                      "run": str(referenced)}]}), encoding="utf-8")
    running = task(weights / "teacher-tasks" / ("a" * 32), "running", rounds=2)
    completed = task(weights / "selfplay-tasks" / ("b" * 32), "completed",
                     rounds=3, stamp=2000)
    stopped = task(weights / "transition-tasks" / ("c" * 32), "stopped",
                   rounds=1)
    bundle(weights / "value-run", stamp=3000)
    bundle(weights / "value-demo", stamp=4000)
    return {"weights": weights, "evaluations": evaluations,
            "registry": registry, "guard": guard, "running": running,
            "completed": completed, "stopped": stopped,
            "referenced": referenced, "stale": stale}


def rows(tree):
    out = []
    for i in range(tree.topLevelItemCount()):
        group = tree.topLevelItem(i)
        for j in range(group.childCount()):
            item = group.child(j)
            out.append((group.text(0), item.text(0), item))
    return out


def main():
    app = QApplication.instance() or QApplication([sys.argv[0]])
    with tempfile.TemporaryDirectory(prefix="gui-cleanup-") as raw:
        base = Path(raw)
        paths = build(base)
        settings = QSettings(str(base / "client.ini"),
                             QSettings.Format.IniFormat)
        window = gomoku_gui.MainWindow(
            EXE, 50, registry=paths["registry"], weights_root=paths["weights"],
            evaluations=paths["evaluations"], settings=settings)
        window.resize(900, 560)
        window.show()
        pump(app)
        window._show_settings()
        pump(app, 0.1)
        assert window.btn_cleanup_models.isVisible()
        assert window.btn_cleanup_models.isEnabled()
        summary = window.cleanup_summary.text()
        assert "可清理" in summary and "历史数据" in summary, summary
        window.settings_sheet.hide()
        pump(app, 0.05)

        # Busy training refuses to open the dialog.
        window._phase = "training"
        window._open_cleanup()
        assert "任务运行中" in window.error_label.text(), window.error_label.text()
        window._phase = "idle"

        dialog = CleanupDialog(paths["weights"], paths["guard"],
                               paths["evaluations"], loaded_model=None,
                               keep=1, parent=window)
        dialog.show()
        pump(app)
        assert dialog.tabs.count() == 2
        assert [dialog.tabs.tabText(i) for i in range(2)] == ["候选模型", "历史数据"]

        model_rows = rows(dialog.model_tree)
        data_rows = rows(dialog.data_tree)
        assert model_rows and data_rows
        model_checkable = [item for _g, _n, item in model_rows
                           if item.flags() & Qt.ItemIsUserCheckable]
        model_checked = [item for item in model_checkable
                         if item.checkState(0) == Qt.Checked]
        # Old rounds of the completed task are pre-selected.
        assert len(model_checked) == 2, [item.text(0) for item in model_checked]
        data_checkable = [item for _g, _n, item in data_rows
                          if item.flags() & Qt.ItemIsUserCheckable]
        data_checked = [item for item in data_checkable
                        if item.checkState(0) == Qt.Checked]
        # Only the completed task directory is pre-selected: the stopped task
        # is resumable and the stale evaluation is evidence-adjacent.
        assert len(data_checked) == 1, [item.text(0) for item in data_checked]
        assert "可清理" in dialog.model_summary.text()
        assert "可清理" in dialog.data_summary.text()

        original = (QMessageBox.question, QMessageBox.information,
                    QMessageBox.warning)
        QMessageBox.question = staticmethod(
            lambda *a, **k: QMessageBox.StandardButton.Yes)
        QMessageBox.information = staticmethod(lambda *a, **k: None)
        QMessageBox.warning = staticmethod(lambda *a, **k: None)
        reports = []
        dialog.cleaned.connect(lambda count, freed: reports.append((count, freed)))
        try:
            dialog._delete()
        finally:
            (QMessageBox.question, QMessageBox.information,
             QMessageBox.warning) = original
        assert reports and reports[0][0] >= 1, reports
        assert reports[0][1] > 0

        # The default selection removes the completed task (plus its old
        # rounds); running, stopped, evidence, fixture and manual output stay.
        assert not paths["completed"].exists()
        assert paths["running"].exists()
        assert paths["stopped"].exists()
        assert paths["referenced"].exists()
        assert paths["stale"].exists()
        assert (paths["weights"] / "value-run" / "model.gmvalue").is_file()
        assert (paths["weights"] / "value-demo" / "model.gmvalue").is_file()

        # An explicitly ticked stale report can still be removed.
        stale_item = next(item for _g, name, item in rows(dialog.data_tree)
                          if name == "run-stale")
        stale_item.setCheckState(0, Qt.Checked)
        dialog._sync_summary()
        original = (QMessageBox.question, QMessageBox.information,
                    QMessageBox.warning)
        QMessageBox.question = staticmethod(
            lambda *a, **k: QMessageBox.StandardButton.Yes)
        QMessageBox.information = staticmethod(lambda *a, **k: None)
        QMessageBox.warning = staticmethod(lambda *a, **k: None)
        try:
            dialog._delete()
        finally:
            (QMessageBox.question, QMessageBox.information,
             QMessageBox.warning) = original
        assert not paths["stale"].exists()
        assert paths["stopped"].exists()
        assert paths["referenced"].exists()

        window._refresh_cleanup_summary()
        assert "可清理" in window.cleanup_summary.text()
        assert window.combo_model.count() >= 2

        dialog.close()
        window.close()
        app.processEvents()
    print("gui_cleanup_smoke: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
