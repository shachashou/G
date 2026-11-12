#!/usr/bin/env python3
"""Offscreen GUI smoke test for continuous collection (ticket 05).

Starts a tiny real collection from the window, checks the metrics panel and
the pool it wrote, then starts another run and stops it, checking that the UI
returns to idle with nothing staged. Skips (77) when Rapfi is not installed.

Usage: python gui_collect_smoke.py <gomoku_nnue.exe>
"""

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
from sample_pool import SamplePool  # noqa: E402

EXE = sys.argv[1] if len(sys.argv) > 1 else str(
    ROOT / "build" / "engine" / "gomoku_nnue.exe")
TEMP = tempfile.TemporaryDirectory(prefix="gui-collect-")
POOL = Path(TEMP.name) / "pool.sqlite3"

failures = 0


def check(cond, msg):
    global failures
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures += 1


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
    if not rapfi_teacher.check_install(identity).ok:
        print("  SKIP teacher not installed")
        return 77
    app = QApplication.instance() or QApplication([sys.argv[0]])
    window = gomoku_gui.MainWindow(EXE, 100, pool=POOL,
                                  registry=Path(TEMP.name) / "registry",
                                  collection_seed=1)
    window.show()
    pump(app, 0.3)

    window.spin_collect_games.setValue(1)
    window.spin_collect_capacity.setValue(60)
    window.spin_collect_budget.setValue(150)
    window._start_collection()
    check(wait_for(app, lambda: window._preview is not None, 30),
          "collection publishes a live board preview")
    finished = wait_for(app, lambda: "已完成" in window.collect_label.text()
                        or "已停止" in window.collect_label.text(), 120)
    check(finished, "first collection run finished")
    check(window._phase == "idle", "GUI returned to idle after collection")
    check("入池" in window.collect_label.text()
          and "有效/秒" in window.collect_label.text(),
          "metrics show inserts and effective samples per second")
    check(POOL.exists(), "collection created the sample pool")
    with SamplePool(POOL, create=False) as pool:
        stats = pool.stats()
        problems = pool.validate()
    check(stats["pool_samples"] > 0, "collection wrote samples")
    check(stats["staged_games"] == 0, "no staged game is left behind")
    check(problems == [], "collection pool validates")
    check(stats["validation_samples"] > 0, "validation split was frozen")
    check("样本池" in window.collect_pool_label.text(),
          "pool status label was refreshed")

    # Start another run and stop it from the UI.
    window.spin_collect_games.setValue(3)
    window._start_collection()
    started = wait_for(app, lambda: "生成" in window.collect_label.text(), 30)
    check(started, "second run started and reported progress")
    window._stop_collection()
    stopped = wait_for(app, lambda: window._phase == "idle", 90)
    check(stopped, "stop returns the GUI to idle")
    check("已停止" in window.collect_label.text()
          or "已完成" in window.collect_label.text(),
          "stop is reflected in the metrics panel")
    with SamplePool(POOL, create=False) as pool:
        stats = pool.stats()
    check(stats["staged_games"] == 0, "stopping leaves nothing staged")

    # Missing teacher dependency: the start button must fail loudly.
    original = rapfi_teacher.check_install
    try:
        rapfi_teacher.check_install = lambda *a, **k: rapfi_teacher.InstallCheck(
            False, ROOT / "tools" / "rapfi", None,
            ["engine missing (simulated)"])
        window._start_collection()
        check("不可用" in window.error_label.text(),
              "missing teacher is reported in the error area")
        check(window._phase == "idle", "missing teacher does not start a run")
    finally:
        rapfi_teacher.check_install = original

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
