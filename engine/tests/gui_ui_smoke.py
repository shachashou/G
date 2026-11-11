#!/usr/bin/env python3
"""Ticket 13 offscreen acceptance: calm layout, primary actions, disclosure."""
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ("gui", "tools", "trainer")]
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
        window = gomoku_gui.MainWindow(EXE, 50)
        window.resize(1264, 820)
        window.show()
        pump(app)
        style = app.styleSheet()
        for token in ("#F5F5F7", "#007AFF", "#FFFFFF"):
            assert token in style, token
        assert window.tabs.count() == 4, window.tabs.count()
        labels = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        assert labels == ["玩一把", "收集数据", "从0训练 / 自我训练", "测评"], labels
        assert window.header_status.text()
        assert window.findChild(QFrame, "BoardCard") is not None
        scroll = window.findChild(QScrollArea)
        assert scroll is not None
        assert scroll.widget().minimumSizeHint().width() <= scroll.viewport().width(), (
            scroll.widget().minimumSizeHint().width(), scroll.viewport().width())
        assert scroll.horizontalScrollBar().maximum() == 0

        # Advanced material is quiet by default, but remains fully functional.
        assert not window.play_advanced.expanded
        assert not window.train_advanced.expanded
        assert not window.training_task_panel.settings.expanded
        assert not window.history_panel.advanced.expanded
        assert not window.play_advanced.body.isVisible()
        window.play_advanced.toggle.click()
        pump(app, 0.05)
        assert window.play_advanced.body.isVisible()
        assert window.spin_think.isVisible()
        window.spin_think.setValue(777)
        assert window.spin_think.value() == 777

        for index, required in (
                (0, (window.btn_new, window.btn_stop)),
                (1, (window.btn_collect_start, window.btn_collect_pause,
                     window.btn_collect_stop)),
                (2, (window.btn_train_start, window.training_task_panel.btn_start)),
                (3, (window.history_panel.btn_initial,
                     window.history_panel.btn_challenge))):
            window.tabs.setCurrentIndex(index)
            pump(app, 0.05)
            for widget in required:
                assert widget.isVisible(), (index, widget.objectName(), widget.text())
        # One primary action per visible mode: the primary pill is blue, the
        # destructive action is red, and the rest stay quiet.
        assert window.btn_new.objectName() == "Primary"
        assert window.btn_stop.objectName() == "Danger"
        assert window.training_task_panel.btn_start.objectName() == "Primary"
        assert window.history_panel.btn_challenge.objectName() == "Primary"

        window.tabs.setCurrentIndex(2)
        window.training_task_panel.settings.toggle.click()
        pump(app, 0.05)
        assert window.training_task_panel.settings.body.isVisible()
        assert window.training_task_panel.spin_steps.isVisible()
        window.training_task_panel.spin_steps.setValue(33)
        assert window.training_task_panel.spin_steps.value() == 33

        window.close()
        app.processEvents()
    print("gui_ui_smoke: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
