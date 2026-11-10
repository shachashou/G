#!/usr/bin/env python3
"""Offscreen GUI smoke test for the model picker (ticket 07).

Builds the real window without a display, loads a synthetic GMVALUE1 model and
checks what the user sees: the actually loaded model identity, the candidate
(not promoted) status, a real engine move from that model, switching back to
the handcrafted evaluator and the explicit failure path for a corrupt file.

Usage: python gui_model_smoke.py <gomoku_nnue.exe>
Requires QT_QPA_PLATFORM=offscreen.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gui"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "trainer"))

from PySide6.QtWidgets import QApplication  # noqa: E402

import gomoku_gui  # noqa: E402
import value_format as vf  # noqa: E402

EXE = sys.argv[1] if len(sys.argv) > 1 else str(
    ROOT / "build" / "engine" / "gomoku_nnue.exe")
FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        FAILURES.append(message)


def pump(app: QApplication, seconds: float) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)


def write_model(path: Path, model_id: str) -> Path:
    ea = 9
    weights = {
        "W0": [0] * (vf.N_FEATURES * vf.N_ACC),
        "b0": [0] * vf.N_ACC,
        "W1": [0] * (vf.N_ACC * vf.N_H1),
        "b1": [0] * vf.N_H1,
        "W2": [0] * (vf.N_H1 * vf.N_H2),
        "b2": [0] * vf.N_H2,
        "W3": [0] * (vf.N_H2 * vf.N_OUT),
        "b3": [120],
        "ea": ea,
        "act_max_int": int(round(8.0 * (1 << ea))),
    }
    metadata = {
        "model_id": model_id,
        "arch": [vf.N_FEATURES, vf.N_ACC, vf.N_H1, vf.N_H2, vf.N_OUT],
        "rule": "renju",
        "feature_spec": "absolute-color-450",
        "value_contract": "black-absolute-tanh-v1",
        "label_versions": ["gui-smoke-v1"],
        "training": {"steps": 1},
    }
    path.write_bytes(vf.pack_value_weights(weights, metadata, ea=ea))
    return path


def wait_for_history(window, app: QApplication, count: int) -> bool:
    deadline = time.time() + 20
    while time.time() < deadline:
        pump(app, 0.05)
        state = window._state
        if state is not None and len(state.get("history", [])) >= count:
            return True
        if window._phase not in ("thinking", "idle"):
            return False
    return False


def main() -> int:
    app = QApplication.instance() or QApplication([sys.argv[0]])
    with tempfile.TemporaryDirectory(prefix="gomoku_gui_model_") as raw:
        tmp = Path(raw)
        model = write_model(tmp / "smoke.gmvalue", "gui-model-smoke")
        corrupt = tmp / "corrupt.gmvalue"
        corrupt.write_bytes(model.read_bytes()[:-1])

        print("[startup with a model]")
        window = gomoku_gui.MainWindow(EXE, 100, model=str(model))
        window.show()
        pump(app, 0.4)
        text = window.eval_label.text()
        check("GMVALUE1" in text, "GUI shows the GMVALUE1 evaluation backend")
        check("gui-model-smoke" in text, "GUI shows the actually loaded model id")
        check("450-1024-64-32-1" in text, "GUI shows the model architecture")
        check("候选试玩" in text and "未晋级" in text,
              "GUI marks the model as an unpromoted candidate")
        check(window._about.get("value_model_loaded") is True,
              "window's executor reports the value model as loaded")
        check(window.combo_model.currentData() == str(model),
              "model picker selects the loaded file")

        print("[real engine move with the model]")
        window.radio_white.setChecked(True)
        window._new_game()
        check(wait_for_history(window, app, 1),
              "engine played a first move with the value model")
        check(window._phase in ("idle", "thinking"),
              "window is responsive after the engine move")
        window._stop_game()

        print("[switch to handcrafted and back]")
        check(window._apply_model_path(None) is True,
              "switching to the handcrafted evaluator succeeds")
        check("手工威胁评估" in window.eval_label.text(),
              "GUI shows the handcrafted evaluation after unloading")
        window._stop_game()
        pump(app, 0.1)
        check(window._apply_model_path(str(model)) is True,
              "switching back to the model succeeds")
        check("GMVALUE1" in window.eval_label.text(),
              "GUI shows the value model again")
        window._stop_game()
        pump(app, 0.1)

        print("[corrupt model is rejected without pretending]")
        check(window._apply_model_path(str(corrupt)) is False,
              "corrupt model is rejected")
        check("模型加载失败" in window.error_label.text(),
              "GUI reports the load failure")
        check("GMVALUE1" in window.eval_label.text(),
              "the previous valid model stays in use and visible")

        window.close()
        pump(app, 0.2)

        print("[startup with a corrupt model]")
        fallback = gomoku_gui.MainWindow(EXE, 100, model=str(corrupt))
        fallback.show()
        pump(app, 0.4)
        check("模型加载失败" in fallback.error_label.text(),
              "startup load failure is visible in the window")
        check("手工威胁评估" in fallback.eval_label.text(),
              "the fallback evaluation is named honestly")
        check("GMVALUE1" not in fallback.eval_label.text(),
              "a failed model is never shown as loaded")
        fallback.close()
        pump(app, 0.2)

    if FAILURES:
        print(f"\n{len(FAILURES)} GUI model check(s) failed")
        return 1
    print("gui_model_smoke: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
