#!/usr/bin/env python3
"""Offscreen GUI smoke test for the teacher panel (ticket 04).

Checks what the user sees: the teacher identity, the labelling result with its
raw score/filter reason and the valid-sample counter, and the explicit failure
path when the teacher dependency is missing (no fabricated labels).

Usage: python gui_teacher_smoke.py <gomoku_nnue.exe>
Requires QT_QPA_PLATFORM=offscreen and an installed Rapfi (else exit 77).
"""
from __future__ import annotations

import sys
import time
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gui"))
sys.path.insert(0, str(ROOT / "tools"))

from PySide6.QtWidgets import QApplication  # noqa: E402

import gomoku_gui  # noqa: E402
import rapfi_teacher  # noqa: E402
from teacher_store import TeacherStore  # noqa: E402

EXE = sys.argv[1] if len(sys.argv) > 1 else str(
    ROOT / "build" / "engine" / "gomoku_nnue.exe")
TEMP = tempfile.TemporaryDirectory(prefix="gui-teacher-")
STORE = Path(TEMP.name) / "samples.sqlite3"

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


def wait_for(app, predicate, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def main():
    identity = rapfi_teacher.load_lock()
    install = rapfi_teacher.check_install(identity)
    if not install.ok:
        print(f"  SKIP teacher not installed: {install.summary()}")
        return 77
    before = 0
    if STORE.exists():
        with TeacherStore(STORE, create=False) as store:
            before = store.counts()["accepted"]

    app = QApplication.instance() or QApplication([sys.argv[0]])
    window = gomoku_gui.MainWindow(EXE, 100, teacher_store=STORE,
                                  registry=Path(TEMP.name) / "registry")
    window.show()
    pump(app, 0.3)

    check("Rapfi" in window.teacher_label.text(),
          "teacher panel shows the teacher identity")
    check(identity.release in window.teacher_label.text()
          and "250615" in window.teacher_label.text(),
          "teacher identity names the pinned release")

    state = window._client.new_game("renju")
    for x, y in [(7, 7), (7, 8), (8, 7), (8, 8)]:
        state = window._client.play(x, y)
        check(state.get("ok") is True, f"test move ({x},{y}) played")
    window._state = state
    window._phase = "idle"
    window._refresh()

    window._label_teacher()
    labelled = wait_for(app, lambda: "有效" in window.teacher_result.text(), 30)
    check(labelled, "teacher labelling finished and updated the panel")
    text = window.teacher_result.text()
    check("已保存" in text or "未入库" in text,
          "panel shows whether the sample was stored")
    check("有效" in text and "请求" in text,
          "panel shows the valid-sample / request counts")
    check(window._phase == "idle", "GUI returns to the idle phase")

    with TeacherStore(STORE, create=False) as store:
        counts = store.counts()
    check(counts["accepted"] >= before + 1,
          "an accepted sample was persisted to the store")
    check(counts["total"] >= counts["accepted"], "request counter is consistent")

    # Missing dependency: the panel must fail loudly and save nothing.
    original_check = rapfi_teacher.check_install
    try:
        rapfi_teacher.check_install = lambda *a, **k: rapfi_teacher.InstallCheck(
            False, ROOT / "tools" / "rapfi", None,
            ["engine missing (simulated)"])
        window._phase = "idle"
        window._label_teacher()
        pumped = wait_for(app, lambda: "不可用" in window.error_label.text(), 5)
        check(pumped, "missing teacher is reported in the error area")
        with TeacherStore(STORE, create=False) as store:
            after = store.counts()
        check(after["total"] == counts["total"],
              "no sample is written when the teacher is missing")
    finally:
        rapfi_teacher.check_install = original_check

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
