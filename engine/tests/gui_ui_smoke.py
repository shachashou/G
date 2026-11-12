#!/usr/bin/env python3
"""Ticket 13/14 offscreen acceptance: calm three-mode layout and disclosure."""
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ("gui", "tools", "trainer")]
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QFrame, QScrollArea
import gomoku_gui

EXE = str(Path(sys.argv[1]).resolve()) if len(sys.argv) > 1 else str(
    ROOT / "build/engine/gomoku_nnue.exe")


def pump(app, seconds=0.2):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)


def main():
    app = QApplication.instance() or QApplication([sys.argv[0]])
    with tempfile.TemporaryDirectory(prefix="gui-ui-") as raw:
        ini = Path(raw) / "client.ini"
        settings = QSettings(str(ini), QSettings.Format.IniFormat)
        window = gomoku_gui.MainWindow(EXE, 50, settings=settings)
        window.resize(1264, 820)
        window.show()
        pump(app)
        style = app.styleSheet()
        for token in ("#F5F5F7", "#007AFF", "#FFFFFF"):
            assert token in style, token
        assert window.tabs.count() == 3, window.tabs.count()
        labels = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        assert labels == ["对弈", "训练", "测评"], labels
        assert window.header_status.text()
        assert window.combo_model.count() >= 2
        assert window.findChild(QFrame, "BoardCard") is not None
        scroll = window.findChild(QScrollArea)
        assert scroll is not None
        assert scroll.widget().minimumSizeHint().width() <= scroll.viewport().width(), (
            scroll.widget().minimumSizeHint().width(), scroll.viewport().width())
        assert scroll.horizontalScrollBar().maximum() == 0

        # Main mode: one primary action and no advanced form.
        assert window.btn_new.isVisible()
        assert window.btn_new.objectName() == "Primary"
        assert window.btn_stop.objectName() == "Danger"
        assert not window.play_advanced.body.isVisible()

        # The board collapses to hand its width to the panel; the preference
        # survives the session and the shortcut mirrors the button.
        assert window.btn_board.isVisible()
        assert window.btn_board.text() == "隐藏棋盘"
        assert window.btn_board.toolTip()
        assert window._board_shortcut.key().toString() == "Ctrl+B"
        window.btn_board.click()
        pump(app, 0.05)
        assert not window.board_card.isVisible()
        assert window.btn_board.text() == "显示棋盘"
        assert window._settings.value(
            "window/board_visible", True, type=bool) is False
        window.btn_board.click()
        pump(app, 0.05)
        assert window.board_card.isVisible()
        assert window.btn_board.text() == "隐藏棋盘"
        assert window._settings.value(
            "window/board_visible", True, type=bool) is True

        # The collapsed board is remembered for the next launch.
        window.btn_board.click()
        pump(app, 0.05)
        window._settings.sync()
        second = gomoku_gui.MainWindow(
            EXE, 50, settings=QSettings(str(ini),
                                        QSettings.Format.IniFormat))
        try:
            assert not second.board_card.isVisible()
            assert second.btn_board.text() == "显示棋盘"
        finally:
            second.close()
            app.processEvents()
        window.btn_board.click()
        pump(app, 0.05)
        assert window.board_card.isVisible()

        # Training has one internal three-way switch and one visible card.
        window.tabs.setCurrentIndex(1)
        pump(app, 0.05)
        assert window.training_mode.buttons["collect"].isVisible()
        assert window.collection_card.isVisible()
        assert not window.training_card.isVisible()
        window.training_mode.set_current("selfplay")
        window._training_mode_changed("selfplay")
        pump(app, 0.05)
        assert window.training_task_panel.isVisible()
        assert not window.collection_card.isVisible()
        assert window.training_task_panel.btn_start.objectName() == "Primary"
        assert not window.training_task_panel.settings.body.isVisible()
        assert "晋级条件" in window.training_task_panel.eval_hint.text()

        # A numerical export rejection is explained, not just a code.
        window.training_task_panel.show_state({
            "status": "completed", "data": {"phase": "stopped"},
            "trainer": {"phase": "exported"},
            "evaluation": {"phase": "retained_incumbent"},
            "games": 3000, "mode": "selfplay",
            "rounds": [{"status": "numerical_rejected",
                        "training": {"steps": 600, "export": {"fidelity": {
                            "float_vs_integer": {"p95_abs": 0.035},
                            "value_tolerance": {"p95_abs": 0.03}}}}}],
            "training_sources": ["selfplay_result"], "source_weights": {},
            "resources": {"device": "cuda"}, "reason": "data_limit",
            "directory": "x", "current": {"version": "v1"}})
        assert "导出保真度未通过" in window.training_task_panel.status.text()
        assert "p95 3.50% > 3.00%" in window.training_task_panel.metrics.toPlainText()

        # The search horizon is visible after every engine move.
        window._show_search_summary({"depth": 6, "nodes": 237000, "score": 12})
        assert "深度 6 层" in window.search_label.text()

        # Task budget survives a restart with no explicit save: edits persist
        # after a short debounce, so a 3000-game cap cannot reset to 1000.
        panel = window.training_task_panel
        panel.spin_games.setValue(3000)
        panel.spin_steps.setValue(321)
        panel.spin_lr.setValue(0.0005)
        panel.combo_mode.setCurrentIndex(panel.combo_mode.findData("selfplay"))
        panel.edit_output.setText(str(Path(raw) / "custom-tasks"))
        panel._queue_task_settings_save()
        pump(app, 0.5)
        assert int(settings.value("training/games")) == 3000
        from training_task_panel import TrainingTaskPanel
        clone = TrainingTaskPanel(
            EXE, window._pool_path, window._registry.root, window._evaluations,
            ROOT / "weights/teacher-tasks", lambda: True, settings=settings)
        try:
            assert clone.spin_games.value() == 3000, clone.spin_games.value()
            assert clone.spin_steps.value() == 321, clone.spin_steps.value()
            assert abs(clone.spin_lr.value() - 0.0005) < 1e-9
            assert clone.combo_mode.currentData() == "selfplay"
            assert clone.edit_output.text() == str(Path(raw) / "custom-tasks")
        finally:
            clone.timer.stop()
            clone.close()
        panel.spin_games.setValue(1000)
        panel.spin_steps.setValue(200)
        panel.spin_lr.setValue(0.001)
        panel.combo_mode.setCurrentIndex(panel.combo_mode.findData("teacher"))
        panel._save_task_settings()

        # Evaluation keeps one primary action (登记 v0 is enabled without an
        # incumbent; 开始测评 is disabled in that state).
        window.tabs.setCurrentIndex(2)
        pump(app, 0.05)
        assert window.history_panel.btn_initial.isVisible()
        assert window.history_panel.btn_initial.objectName() == "Primary"
        assert not window.history_panel.advanced.body.isVisible()

        # Everything advanced is in the right-side sheet, not the main page.
        window.btn_settings.click()
        pump(app, 0.1)
        assert window.settings_sheet.visible()
        assert window.play_advanced.body.isVisible()
        assert window.spin_think.isVisible()
        window.spin_think.setValue(777)
        assert window.spin_think.value() == 777
        window.settings_sheet.show("train")
        pump(app, 0.05)
        assert window.train_advanced.body.isVisible()
        assert window.spin_train_steps.isVisible()
        window.settings_sheet.hide()
        pump(app, 0.05)
        assert not window.settings_sheet.visible()

        window.close()
        app.processEvents()
    print("gui_ui_smoke: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
