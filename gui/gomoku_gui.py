#!/usr/bin/env python3
"""Desktop client for a single legal Renju game (ticket 02).

All rules come from the C++ authoritative executor via tools/executor.py:
complete legal moves, forbidden-move reasons, move results and the terminal
result are never computed in this GUI. Engine thinking runs on a background
thread so the window stays responsive; closing or stopping the game kills
the engine subprocess.
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QMainWindow, QMessageBox, QPushButton, QRadioButton, QSpinBox,
    QVBoxLayout, QWidget,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from executor import ExecutorClient, ExecutorError  # noqa: E402

SIZE = 15
CELL = 36
MARGIN = 30
STONE_R = 15
BOARD_PX = MARGIN * 2 + CELL * (SIZE - 1)

RESULT_TEXT = {
    "ongoing": "对局进行中",
    "black_win": "黑胜",
    "white_win": "白胜",
    "draw": "和棋",
}
REASON_TEXT = {
    "legal": "合法",
    "out_of_bounds": "坐标越界",
    "occupied": "该位置已有棋子",
    "forbidden_overline": "禁手：长连",
    "forbidden_double_four": "禁手：四四",
    "forbidden_double_three": "禁手：三三",
    "game_over": "对局已结束",
}


def px(x: int) -> int:
    return MARGIN + x * CELL


def py(y: int) -> int:
    return MARGIN + y * CELL


class BoardWidget(QWidget):
    moveClicked = Signal(int, int)

    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(BOARD_PX, BOARD_PX)
        self.setCursor(Qt.PointingHandCursor)
        self.cells: list[int] = [0] * (SIZE * SIZE)
        self.legal: set[tuple[int, int]] = set()
        self.forbidden: dict[tuple[int, int], str] = {}
        self.last_move: tuple[int, int] | None = None
        self.markers = True

    def set_position(self, cells, legal, forbidden, last_move, markers) -> None:
        self.cells = cells
        self.legal = legal
        self.forbidden = forbidden
        self.last_move = last_move
        self.markers = markers
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(221, 183, 124))

        painter.setPen(QPen(QColor(59, 42, 24), 1))
        for i in range(SIZE):
            painter.drawLine(px(0), py(i), px(SIZE - 1), py(i))
            painter.drawLine(px(i), py(0), px(i), py(SIZE - 1))
        painter.setBrush(QColor(59, 42, 24))
        painter.setPen(QPen(Qt.NoPen))
        for sx, sy in ((3, 3), (11, 3), (3, 11), (11, 11), (7, 7)):
            painter.drawEllipse(px(sx) - 3, py(sy) - 3, 6, 6)

        if self.markers:
            painter.setPen(QPen(Qt.NoPen))
            painter.setBrush(QColor(0, 150, 0, 110))
            for x, y in self.legal:
                painter.drawEllipse(px(x) - 4, py(y) - 4, 8, 8)
            painter.setPen(QPen(QColor(190, 0, 0, 170), 2))
            for x, y in self.forbidden:
                painter.drawLine(px(x) - 4, py(y) - 4, px(x) + 4, py(y) + 4)
                painter.drawLine(px(x) - 4, py(y) + 4, px(x) + 4, py(y) - 4)

        for m, c in enumerate(self.cells):
            if c == 0:
                continue
            x, y = m % SIZE, m // SIZE
            painter.setBrush(QColor(17, 17, 17) if c == 1
                             else QColor(250, 250, 250))
            painter.setPen(QPen(QColor(60, 60, 60), 1))
            painter.drawEllipse(px(x) - STONE_R, py(y) - STONE_R,
                                2 * STONE_R, 2 * STONE_R)

        if self.last_move is not None:
            x, y = self.last_move
            painter.setPen(QPen(QColor(224, 32, 32), 2))
            painter.setBrush(QBrush())
            painter.drawRect(px(x) - STONE_R - 2, py(y) - STONE_R - 2,
                             2 * STONE_R + 4, 2 * STONE_R + 4)
        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        pos = event.position()
        x = round((pos.x() - MARGIN) / CELL)
        y = round((pos.y() - MARGIN) / CELL)
        if 0 <= x < SIZE and 0 <= y < SIZE:
            if ((pos.x() - px(x)) ** 2 + (pos.y() - py(y)) ** 2
                    <= (CELL * 0.48) ** 2):
                self.moveClicked.emit(x, y)


class MainWindow(QMainWindow):
    def __init__(self, exe: str, think_ms: int) -> None:
        super().__init__()
        self.setWindowTitle("GomokuNNUE · 有禁手五子棋")
        self._exe = exe
        self._think_ms = think_ms
        self._client = ExecutorClient(exe)
        self._about = self._client.about()
        self._state: dict | None = None
        self._phase = "idle"  # idle | thinking | game_over
        self._player_color = "black"
        self._generation = 0
        self._queue: queue.Queue = queue.Queue()
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_queue)
        self._timer.start(40)
        self._new_game()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)
        self.board = BoardWidget()
        self.board.moveClicked.connect(self._on_player_move)
        root.addWidget(self.board)

        panel = QVBoxLayout()

        setup = QGroupBox("对局设置")
        sv = QVBoxLayout(setup)
        colors = QHBoxLayout()
        self.radio_black = QRadioButton("我执黑（先手）")
        self.radio_white = QRadioButton("我执白（后手）")
        self.radio_black.setChecked(True)
        colors.addWidget(self.radio_black)
        colors.addWidget(self.radio_white)
        sv.addLayout(colors)

        btns = QHBoxLayout()
        self.btn_new = QPushButton("新局")
        self.btn_new.clicked.connect(self._new_game)
        self.btn_stop = QPushButton("停止对局")
        self.btn_stop.clicked.connect(self._stop_game)
        btns.addWidget(self.btn_new)
        btns.addWidget(self.btn_stop)
        sv.addLayout(btns)

        self.chk_markers = QCheckBox("显示合法落点 / 禁手原因")
        self.chk_markers.setChecked(True)
        self.chk_markers.toggled.connect(self._on_markers_toggled)
        sv.addWidget(self.chk_markers)

        think = QHBoxLayout()
        think.addWidget(QLabel("引擎思考(ms)"))
        self.spin_think = QSpinBox()
        self.spin_think.setRange(100, 30000)
        self.spin_think.setValue(self._think_ms)
        think.addWidget(self.spin_think)
        sv.addLayout(think)
        panel.addWidget(setup)

        status = QGroupBox("状态")
        stv = QVBoxLayout(status)
        if self._about.get("nnue_loaded"):
            eval_text = ("评估方式：NNUE（"
                         + str(self._about.get("nnue_backend")) + "，"
                         + str(self._about.get("nnue_path")) + "）")
        else:
            eval_text = "评估方式：手工威胁评估（未加载已训练 NNUE 模型）"
        self.eval_label = QLabel(eval_text)
        self.eval_label.setWordWrap(True)
        self.turn_label = QLabel("行棋方：-")
        self.status_label = QLabel("")
        self.result_label = QLabel("对局进行中")
        self.replay_label = QLabel("")
        self.replay_label.setWordWrap(True)
        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #B00020;")
        for widget in (self.eval_label, self.turn_label, self.status_label,
                       self.result_label, self.replay_label, self.error_label):
            stv.addWidget(widget)
        panel.addWidget(status)

        record = QGroupBox("着法记录")
        rv = QVBoxLayout(record)
        self.moves_list = QListWidget()
        rv.addWidget(self.moves_list)
        panel.addWidget(record)

        root.addLayout(panel)
        self.setCentralWidget(central)
        self.resize(BOARD_PX + 330, BOARD_PX + 40)

    # ---- game flow -----------------------------------------------------

    def _new_game(self) -> None:
        self._interrupt()
        self._player_color = "black" if self.radio_black.isChecked() else "white"
        try:
            response = self._client.new_game("renju")
        except ExecutorError as exc:
            self._show_error(str(exc))
            return
        if not response.get("ok"):
            self._show_error(str(response))
            return
        self._state = response
        self._phase = "idle"
        self.replay_label.setText("")
        self.error_label.setText("")
        self.status_label.setText("新局开始：黑先")
        self._refresh()
        self._maybe_engine_move()

    def _stop_game(self) -> None:
        self._interrupt()
        self._phase = "idle"
        self._state = None
        self.board.set_position([0] * (SIZE * SIZE), set(), {}, None, False)
        self.moves_list.clear()
        self.turn_label.setText("行棋方：-")
        self.result_label.setText("已停止")
        self.replay_label.setText("")
        self.error_label.setText("")
        self.status_label.setText("已停止；点击「新局」重新开始")

    def _interrupt(self) -> None:
        """Kill an in-flight engine process without waiting on its lock."""
        if self._phase != "thinking":
            return
        self._generation += 1
        self._client.close()
        self._client = ExecutorClient(self._exe)
        self._about = self._client.about()
        self._phase = "idle"

    def _maybe_engine_move(self) -> None:
        state = self._state
        if state is None:
            return
        if state["result"] != "ongoing":
            self._phase = "game_over"
            self._on_game_over()
            return
        if state["side_to_move"] != self._player_color:
            self._start_thinking()
        else:
            self._phase = "idle"
            self.status_label.setText("请落子")
            self._refresh()

    def _start_thinking(self) -> None:
        self._phase = "thinking"
        self.status_label.setText("引擎思考中…")
        self.error_label.setText("")
        client = self._client
        generation = self._generation
        movetime = self.spin_think.value()

        def worker() -> None:
            try:
                response = client.think(movetime_ms=movetime)
                self._queue.put(("think", generation, response))
            except ExecutorError as exc:
                self._queue.put(("error", generation, str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self) -> None:
        while True:
            try:
                kind, generation, payload = self._queue.get_nowait()
            except queue.Empty:
                return
            if generation != self._generation:
                continue  # a stop/new-game interrupted this thinking step
            if kind == "think":
                self._apply_engine_move(payload)
            else:
                self._phase = "idle"
                self._show_error(payload)

    def _apply_engine_move(self, response: dict) -> None:
        if not response.get("ok"):
            self._phase = "idle"
            self._show_error("引擎选点失败：" + str(response.get("error", response)))
            return
        x, y = response["move"]
        try:
            result = self._client.play(x, y)
        except ExecutorError as exc:
            self._phase = "idle"
            self._show_error(str(exc))
            return
        if not result.get("ok"):
            self._phase = "idle"
            reason = result.get("reason", "")
            self._show_error("引擎落子被拒：" + REASON_TEXT.get(reason, reason))
            return
        self._state = result
        self._refresh()
        self._maybe_engine_move()

    def _on_player_move(self, x: int, y: int) -> None:
        if self._phase != "idle" or self._state is None:
            return
        if self._state["result"] != "ongoing":
            return
        if self._state["side_to_move"] != self._player_color:
            return
        try:
            response = self._client.play(x, y)
        except ExecutorError as exc:
            self._show_error(str(exc))
            return
        if not response.get("ok"):
            reason = response.get("reason", "")
            self._show_error("落子被拒：" + REASON_TEXT.get(reason, reason))
            return
        self._state = response
        self.error_label.setText("")
        self._refresh()
        self._maybe_engine_move()

    def _on_game_over(self) -> None:
        state = self._state
        moves = [list(m) for m in state.get("history", [])]
        try:
            replayed = self._client.replay(moves, "renju")
            consistent = (
                replayed.get("ok") is True
                and replayed.get("result") == state["result"]
                and replayed.get("cells") == state["cells"]
            )
            if consistent:
                self.replay_label.setText(
                    "重放校验：一致（同一执行器重放 " + str(len(moves)) + " 手）")
            else:
                self.replay_label.setText(
                    "重放校验：不一致 / 失败：" + str(replayed.get("error", replayed)))
        except ExecutorError as exc:
            self.replay_label.setText("重放校验失败：" + str(exc))
        self.status_label.setText("对局结束，可查看着法记录或开新局")
        self._refresh()

    def _on_markers_toggled(self, _checked) -> None:
        self._refresh()

    def _refresh(self) -> None:
        state = self._state
        if state is None:
            return
        legal: set[tuple[int, int]] = set()
        forbidden: dict[tuple[int, int], str] = {}
        # Never touch the executor while a think request holds its lock;
        # that would freeze the UI, which must stay responsive.
        if (self.chk_markers.isChecked() and state["result"] == "ongoing"
                and self._phase != "thinking"):
            try:
                lm = self._client.legal_moves()
                legal = {tuple(m) for m in lm.get("moves", [])}
                forbidden = {(f["x"], f["y"]): f["reason"]
                             for f in lm.get("forbidden", [])}
            except ExecutorError as exc:
                self._show_error(str(exc))
        last = tuple(state["last_move"]) if state.get("last_move") else None
        self.board.set_position(state["cells"], legal, forbidden, last,
                                self.chk_markers.isChecked())
        side = "黑" if state["side_to_move"] == "black" else "白"
        who = "（你）" if state["side_to_move"] == self._player_color else "（引擎）"
        self.turn_label.setText(f"行棋方：{side}{who}")
        self.moves_list.clear()
        for i, m in enumerate(state.get("history", [])):
            color = "黑" if i % 2 == 0 else "白"
            self.moves_list.addItem(f"{i + 1:3d}. {color} ({m[0]},{m[1]})")
        self.moves_list.scrollToBottom()
        if state["result"] == "ongoing":
            self.result_label.setText("对局进行中")
        else:
            self.result_label.setText(
                "结果：" + RESULT_TEXT.get(state["result"], state["result"]))

    def _show_error(self, message: str) -> None:
        self.error_label.setText("错误：" + message)

    def closeEvent(self, event) -> None:
        self._generation += 1
        self._timer.stop()
        self._client.close()
        event.accept()


def resolve_engine(explicit: str | None) -> str | None:
    if explicit:
        return str(Path(explicit).resolve())
    env = os.environ.get("GOMOKU_ENGINE")
    if env:
        return env
    for relative in ("build/engine/gomoku_nnue.exe",
                     "build/engine/Release/gomoku_nnue.exe"):
        candidate = ROOT / relative
        if candidate.is_file():
            return str(candidate)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="GomokuNNUE 本地人机对弈")
    parser.add_argument("--engine", help="引擎可执行文件路径")
    parser.add_argument("--movetime", type=int, default=1000,
                        help="引擎每步思考毫秒数（界面可调整）")
    args = parser.parse_args()
    app = QApplication([sys.argv[0]])
    exe = resolve_engine(args.engine)
    if exe is None or not Path(exe).is_file():
        QMessageBox.critical(
            None, "GomokuNNUE",
            "未找到引擎可执行文件。请先按 README 用 venv 的 cmake + Ninja 构建，"
            "或通过 --engine / GOMOKU_ENGINE 指定路径。")
        return 1
    window = MainWindow(exe, args.movetime)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
