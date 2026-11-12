#!/usr/bin/env python3
"""Offscreen acceptance: evaluation candidate guard and retainer explanation."""

import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ("gui", "tools", "trainer")]
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
import gomoku_gui
from history_panel import HistoryPanel
from model_registry import ModelRegistry

EXE = str(Path(sys.argv[1]).resolve()) if len(sys.argv) > 1 else str(
    ROOT / "build/engine/gomoku_nnue.exe")


def pump(app, seconds=0.05):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)


def main():
    app = QApplication.instance() or QApplication([sys.argv[0]])
    with tempfile.TemporaryDirectory(prefix="gui-history-guard-") as raw:
        base = Path(raw)
        model = base / "registry" / "versions" / "v1-abcd" / "model.gmvalue"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"fake gvalue bytes")
        sha = hashlib.sha256(model.read_bytes()).hexdigest()
        registry = base / "registry"
        (registry / "state.json").write_text(json.dumps({
            "format": "GMVALUE-REGISTRY1", "current": "v1",
            "versions": [
                {"version": "v0", "model": "versions/v0/model.gmvalue",
                 "sha256": "0" * 64, "model_id": "v0"},
                {"version": "v1", "model": "versions/v1-abcd/model.gmvalue",
                 "sha256": sha, "model_id": "demo"}]}),
            encoding="utf-8")
        other = base / "candidates" / "round-00000" / "model.gmvalue"
        other.parent.mkdir(parents=True)
        other.write_bytes(b"a different candidate")

        outputs = base / "evaluations"
        panel = HistoryPanel(EXE, registry, outputs, lambda: True)
        panel.show()
        pump(app)

        problem = panel.candidate_problem(str(model))
        assert problem is not None
        assert "当前正式版本 v1" in problem, problem
        assert "不能自己挑战自己" in problem
        assert panel.candidate_problem(str(other)) is None

        # The gate hint spells out the score rate a challenge must beat, so a
        # 51% baseline setting is visibly not "win 51% and promote".
        from history_evaluation import required_score_rate
        assert abs(required_score_rate(64) - 0.65297) < 1e-4
        assert abs(required_score_rate(128) - 0.60817) < 1e-4
        assert required_score_rate(1) == 1.0
        assert "65.3%" in panel.gate_hint.text(), panel.gate_hint.text()
        panel.spin_threshold.setValue(0.51)
        assert "66.3%" in panel.gate_hint.text(), panel.gate_hint.text()
        panel.spin_threshold.setValue(0.5)
        panel.spin_pairs.setValue(128)
        assert "60.8%" in panel.gate_hint.text(), panel.gate_hint.text()
        panel.spin_pairs.setValue(64)
        assert "置信下界" in panel.gate_hint.text()

        # The pre-check runs before any worker starts and leaves no run dir.
        panel.edit_candidate.setText(str(model))
        panel.start(False)
        assert panel.thread is None
        assert "不能自己挑战自己" in panel.status.text(), panel.status.text()
        assert not outputs.exists() or not any(outputs.iterdir())

        # A retained candidate explains itself instead of a bare status code.
        panel.show_report({
            "status": "completed", "run": str(base), "publication": None,
            "statistics": {"v1": {
                "wins": 40, "draws": 0, "losses": 24, "valid_games": 64,
                "strict_win_rate": 0.625, "score_rate": 0.625,
                "paired_score_rate": 0.625, "one_sided_lower": 0.472,
                "one_sided_upper": 0.778, "threshold": 0.5,
                "completed_pairs": 64, "passed": False,
                "conclusion": "insufficient_evidence", "anomalies": {}}},
        })
        report = panel.report.toPlainText()
        assert "证据不足" in report, report
        assert "增加开局对数" in report, report
        assert "保留原正式版本" in panel.status.text(), panel.status.text()

        # The capsule must not push the current formal version into the
        # evaluation candidate field.
        window = gomoku_gui.MainWindow(
            EXE, 50, registry=registry,
            settings=QSettings(str(base / "client.ini"),
                               QSettings.Format.IniFormat))
        try:
            window._registry = ModelRegistry(registry)
            window._model = str(model)
            window._sync_history_candidate()
            assert window.history_panel.edit_candidate.text() == ""
            window._model = str(other)
            window._sync_history_candidate()
            assert window.history_panel.edit_candidate.text() == str(other)
        finally:
            window.close()
            app.processEvents()

        panel.close()
        app.processEvents()
    print("gui_history_guard_smoke: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
