#!/usr/bin/env python3
"""Offscreen GUI smoke test for the VCF panel (ticket 03).

Constructs the real window (no display needed), rebuilds a proven VCF position
through the executor, runs the "算杀检查" action on its worker thread and
checks what the user would see: status, first move and variation.

Usage: python gui_vcf_smoke.py <gomoku_nnue.exe>
Requires QT_QPA_PLATFORM=offscreen.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gui"))
sys.path.insert(0, str(ROOT / "tools"))

from PySide6.QtWidgets import QApplication  # noqa: E402

import gomoku_gui  # noqa: E402

EXE = sys.argv[1] if len(sys.argv) > 1 else str(
    ROOT / "build" / "engine" / "gomoku_nnue.exe")

SIZE = 15
failures = 0


def check(cond, msg):
    global failures
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures += 1


def cells_from(blacks, whites):
    cells = [0] * (SIZE * SIZE)
    for x, y in blacks:
        cells[y * SIZE + x] = 1
    for x, y in whites:
        cells[y * SIZE + x] = 2
    return cells


def pump(app, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)


def main():
    app = QApplication.instance() or QApplication([sys.argv[0]])
    window = gomoku_gui.MainWindow(EXE, 100)
    window.show()
    pump(app, 0.3)

    check("算杀" in window.vcf_engine_label.text(),
          "window shows the VCF engine identity")
    check("quintet@" in window.vcf_engine_label.text(),
          "VCF engine identity names the vendored upstream")
    check(window.vcf_label.text().startswith("算杀：未检查"),
          "VCF panel starts idle")

    blacks = [(7, 7), (9, 7), (10, 7), (8, 6), (11, 6), (8, 5)]
    whites = [(7, 8), (11, 7), (9, 6), (10, 5), (8, 4), (11, 4)]
    state = window._client.rebuild(cells_from(blacks, whites), "black", "renju")
    check(state.get("ok") is True, "test position rebuilt")
    window._state = state
    window._phase = "idle"
    window._refresh()

    window._check_vcf()
    deadline = time.time() + 20
    while time.time() < deadline and "已证明" not in window.vcf_label.text():
        pump(app, 0.05)
    text = window.vcf_label.text()
    check("已证明" in text, "GUI shows the proven VCF status")
    check("首着" in text and "(8,7)" in text,
          "GUI shows the proven first move")
    check("变化" in text and "(6,7)" in text, "GUI shows the variation")
    check("预算" in text, "GUI shows the budget the query used")
    check(window._phase == "idle", "GUI returns to the idle phase")

    # Stopping a query must not leave the UI stuck: a stop resets the panel.
    window._stop_game()
    check("已停止" in window.vcf_label.text(),
          "stopping the game reports the VCF state as stopped")
    check(window._phase == "idle", "stop leaves the GUI idle")

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
