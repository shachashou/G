#!/usr/bin/env python3
"""Ticket 14 offscreen acceptance: live background board mirroring."""
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ("gui", "tools", "trainer")]
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
import gomoku_gui

EXE = str(Path(sys.argv[1]).resolve()) if len(sys.argv) > 1 else str(
    ROOT / "build/engine/gomoku_nnue.exe")


def pump(app, seconds=0.1):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)


def board_with(stones):
    cells = [0] * 225
    for i, (x, y) in enumerate(stones):
        cells[y * 15 + x] = 1 if i % 2 == 0 else 2
    return cells


def payload(game, stones, side="white", last=(7, 7), ply=None, kind="collect"):
    return {"kind": kind, "cells": board_with(stones),
            "side_to_move": side, "last_move": list(last),
            "ply": ply if ply is not None else len(stones),
            "result": "ongoing", "game": game, "pair": None,
            "engine": None, "model_id": "demo-model", "worker": 0}


def main():
    app = QApplication.instance() or QApplication([sys.argv[0]])
    with tempfile.TemporaryDirectory(prefix="gui-live-") as raw:
        settings = QSettings(str(Path(raw) / "client.ini"),
                             QSettings.Format.IniFormat)
        window = gomoku_gui.MainWindow(EXE, 50, settings=settings)
        window.resize(1264, 820)
        window.show()
        pump(app)
        window._set_background(True)
        assert window.activity_label.text()
        window._on_preview(payload(1, [(7, 7), (8, 8)], side="black",
                                  last=(7, 7), ply=2))
        pump(app)
        assert window.board.preview_mode
        assert "LIVE" in window.board.banner
        assert "采集" in window.board.banner
        assert window.board.cells[7 * 15 + 7] == 1

        # A second worker must not mix into the followed game inside the
        # switch throttle window.
        followed = list(window.board.cells)
        window._on_preview(payload(2, [(0, 0), (1, 1)], last=(0, 0), ply=2))
        assert window.board.cells == followed
        assert window.board.preview_mode

        # Pause freezes and dims the last frame instead of inventing moves.
        window._preview_paused = True
        window._preview_dim = True
        window._render_preview()
        assert window.board.banner == "已暂停"
        assert window.board.dim

        # A player's own in-progress game is not hijacked; LIVE becomes a
        # toggle. Clicking the board is ignored while the mirror is shown.
        state = window._client.state()
        state = dict(state, move_count=1,
                     history=[[7, 7]], last_move=[7, 7],
                     side_to_move="white", result="ongoing")
        window._state = state
        window._preview_paused = False
        window._preview_dim = False
        window._live_forced = False
        window._sync_live_button()
        assert window.btn_live.isVisible()
        assert "LIVE" in window.btn_live.text()

        # Hiding the board removes the mirror affordance without stopping the
        # background task; showing the board brings it back.
        window._toggle_board()
        assert not window.board_card.isVisible()
        assert not window.btn_live.isVisible()
        window._on_preview(payload(1, [(7, 7), (8, 8), (9, 9)], side="black",
                                  last=(9, 9), ply=3))
        assert window._preview["cells"][9 * 15 + 9] == 1
        window._toggle_board()
        assert window.board_card.isVisible()
        assert window.btn_live.isVisible()
        played = []
        window._client.play = lambda x, y: played.append((x, y))
        window._on_player_move(0, 0)
        assert played == []

        window._toggle_live()
        assert window._live_showing()
        assert "返回对局" in window.btn_live.text()
        window._toggle_live()
        assert not window._live_showing()
        assert not window.board.preview_mode

        # Stopping the background task restores the player board and clears
        # the live affordance.
        window._set_background(False)
        window._set_background(True)
        window._live_forced = True
        window._on_preview(payload(3, [(7, 7)], side="black", last=(7, 7),
                                  ply=1))
        assert window.board.preview_mode
        window._set_background(False)
        pump(app)
        assert not window.board.preview_mode
        assert not window.btn_live.isVisible()
        window.close()
        app.processEvents()
    print("gui_live_board_smoke: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
