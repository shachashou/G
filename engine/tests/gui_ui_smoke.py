#!/usr/bin/env python3
"""Ticket 13/14 offscreen acceptance: calm three-mode layout and disclosure."""
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
