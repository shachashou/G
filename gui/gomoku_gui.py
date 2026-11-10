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
import time
import uuid
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox,
    QPushButton, QRadioButton, QSpinBox, QVBoxLayout, QWidget,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "trainer"))
from executor import ExecutorClient, ExecutorError  # noqa: E402
import collect  # noqa: E402
import rapfi_teacher  # noqa: E402
import teacher_label  # noqa: E402
import vcf_tactics  # noqa: E402
from sample_pool import PoolError, SamplePool  # noqa: E402
from teacher_store import TeacherStore  # noqa: E402

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
        self._game_id = uuid.uuid4().hex[:12]
        self._active_teacher = None
        self._collect_stop = None
        self._collect_thread = None
        self._collect_generation = 0
        self._train_stop = None
        self._train_thread = None
        self._train_generation = 0
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

        vcf_row = QHBoxLayout()
        self.btn_vcf = QPushButton("算杀检查（VCF）")
        self.btn_vcf.clicked.connect(self._check_vcf)
        vcf_row.addWidget(self.btn_vcf)
        vcf_row.addWidget(QLabel("节点预算"))
        self.spin_vcf = QSpinBox()
        self.spin_vcf.setRange(100, 5000000)
        self.spin_vcf.setSingleStep(1000)
        self.spin_vcf.setValue(50000)
        vcf_row.addWidget(self.spin_vcf)
        sv.addLayout(vcf_row)

        think = QHBoxLayout()
        think.addWidget(QLabel("引擎思考(ms)"))
        self.spin_think = QSpinBox()
        self.spin_think.setRange(100, 30000)
        self.spin_think.setValue(self._think_ms)
        think.addWidget(self.spin_think)
        sv.addLayout(think)
        panel.addWidget(setup)

        teacher = QGroupBox("教师打标（Rapfi）")
        tv = QVBoxLayout(teacher)
        self.teacher_label = QLabel(self._teacher_status_text())
        self.teacher_label.setWordWrap(True)
        tv.addWidget(self.teacher_label)
        teacher_row = QHBoxLayout()
        self.btn_teacher = QPushButton("打标并保存样本")
        self.btn_teacher.clicked.connect(self._label_teacher)
        teacher_row.addWidget(self.btn_teacher)
        teacher_row.addWidget(QLabel("预算(ms)"))
        self.spin_teacher_budget = QSpinBox()
        self.spin_teacher_budget.setRange(100, 10000)
        self.spin_teacher_budget.setSingleStep(100)
        self.spin_teacher_budget.setValue(800)
        teacher_row.addWidget(self.spin_teacher_budget)
        tv.addLayout(teacher_row)
        self.teacher_result = QLabel("样本：未打标")
        self.teacher_result.setWordWrap(True)
        tv.addWidget(self.teacher_result)
        panel.addWidget(teacher)

        collection = QGroupBox("持续采集")
        cv = QVBoxLayout(collection)
        self.collect_pool_label = QLabel(self._pool_status_text())
        self.collect_pool_label.setWordWrap(True)
        cv.addWidget(self.collect_pool_label)
        collect_row = QHBoxLayout()
        collect_row.addWidget(QLabel("局数"))
        self.spin_collect_games = QSpinBox()
        self.spin_collect_games.setRange(1, 200)
        self.spin_collect_games.setValue(2)
        collect_row.addWidget(self.spin_collect_games)
        collect_row.addWidget(QLabel("容量"))
        self.spin_collect_capacity = QSpinBox()
        self.spin_collect_capacity.setRange(10, 100000)
        self.spin_collect_capacity.setSingleStep(50)
        self.spin_collect_capacity.setValue(200)
        collect_row.addWidget(self.spin_collect_capacity)
        collect_row.addWidget(QLabel("预算(ms)"))
        self.spin_collect_budget = QSpinBox()
        self.spin_collect_budget.setRange(100, 2000)
        self.spin_collect_budget.setSingleStep(100)
        self.spin_collect_budget.setValue(300)
        collect_row.addWidget(self.spin_collect_budget)
        cv.addLayout(collect_row)
        collect_buttons = QHBoxLayout()
        self.btn_collect_start = QPushButton("启动采集")
        self.btn_collect_start.clicked.connect(self._start_collection)
        self.btn_collect_stop = QPushButton("停止采集")
        self.btn_collect_stop.clicked.connect(self._stop_collection)
        self.btn_collect_stop.setEnabled(False)
        collect_buttons.addWidget(self.btn_collect_start)
        collect_buttons.addWidget(self.btn_collect_stop)
        cv.addLayout(collect_buttons)
        self.collect_label = QLabel("采集：未开始")
        self.collect_label.setWordWrap(True)
        cv.addWidget(self.collect_label)
        panel.addWidget(collection)

        training = QGroupBox("价值网络训练")
        tv2 = QVBoxLayout(training)
        pool_row = QHBoxLayout()
        pool_row.addWidget(QLabel("样本池"))
        self.edit_pool = QLineEdit(str(ROOT / "data" / "pool" / "pool.sqlite3"))
        pool_row.addWidget(self.edit_pool)
        tv2.addLayout(pool_row)
        train_row = QHBoxLayout()
        train_row.addWidget(QLabel("步数"))
        self.spin_train_steps = QSpinBox()
        self.spin_train_steps.setRange(1, 100000)
        self.spin_train_steps.setValue(100)
        train_row.addWidget(self.spin_train_steps)
        train_row.addWidget(QLabel("批"))
        self.spin_train_batch = QSpinBox()
        self.spin_train_batch.setRange(1, 512)
        self.spin_train_batch.setValue(16)
        train_row.addWidget(self.spin_train_batch)
        train_row.addWidget(QLabel("lr"))
        self.spin_train_lr = QDoubleSpinBox()
        self.spin_train_lr.setDecimals(5)
        self.spin_train_lr.setRange(0.00001, 1.0)
        self.spin_train_lr.setSingleStep(0.0005)
        self.spin_train_lr.setValue(0.001)
        train_row.addWidget(self.spin_train_lr)
        self.combo_train_loss = QComboBox()
        self.combo_train_loss.addItems(["MSE", "SmoothL1"])
        train_row.addWidget(self.combo_train_loss)
        tv2.addLayout(train_row)
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("输出"))
        self.edit_train_output = QLineEdit(str(ROOT / "weights" / "value-run"))
        out_row.addWidget(self.edit_train_output)
        tv2.addLayout(out_row)
        train_buttons = QHBoxLayout()
        self.btn_train_start = QPushButton("开始训练")
        self.btn_train_start.clicked.connect(self._start_training)
        self.btn_train_stop = QPushButton("停止训练")
        self.btn_train_stop.clicked.connect(self._stop_training)
        self.btn_train_stop.setEnabled(False)
        train_buttons.addWidget(self.btn_train_start)
        train_buttons.addWidget(self.btn_train_stop)
        tv2.addLayout(train_buttons)
        self.train_label = QLabel("训练：未开始")
        self.train_label.setWordWrap(True)
        tv2.addWidget(self.train_label)
        self.train_result = QLabel("")
        self.train_result.setWordWrap(True)
        tv2.addWidget(self.train_result)
        panel.addWidget(training)

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
        self.vcf_label = QLabel("算杀：未检查")
        self.vcf_label.setWordWrap(True)
        self.vcf_engine_label = QLabel(
            "算杀引擎：" + str(self._about.get("vcf_engine", "未加载")))
        self.vcf_engine_label.setWordWrap(True)
        self.result_label = QLabel("对局进行中")
        self.replay_label = QLabel("")
        self.replay_label.setWordWrap(True)
        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #B00020;")
        for widget in (self.eval_label, self.vcf_engine_label, self.turn_label,
                       self.status_label, self.vcf_label, self.result_label,
                       self.replay_label, self.error_label):
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
        if self._phase == "collecting":
            self._show_error("采集运行中，请先停止采集")
            return
        if self._phase == "training":
            self._show_error("训练运行中，请先停止训练")
            return
        self._interrupt()
        self._player_color = "black" if self.radio_black.isChecked() else "white"
        self._game_id = uuid.uuid4().hex[:12]
        self.teacher_result.setText("样本：未打标")
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
        self.vcf_label.setText("算杀：未检查")
        self.status_label.setText("新局开始：黑先")
        self._refresh()
        self._maybe_engine_move()

    def _stop_game(self) -> None:
        if self._phase == "collecting":
            self._show_error("请用「停止采集」结束采集任务")
            return
        if self._phase == "training":
            self._show_error("请用「停止训练」结束训练任务")
            return
        self._interrupt()
        self._phase = "idle"
        self._state = None
        self.board.set_position([0] * (SIZE * SIZE), set(), {}, None, False)
        self.moves_list.clear()
        self.turn_label.setText("行棋方：-")
        self.result_label.setText("已停止")
        self.vcf_label.setText("算杀：已停止（结果未知）")
        self.replay_label.setText("")
        self.error_label.setText("")
        self.status_label.setText("已停止；点击「新局」重新开始")

    def _interrupt(self) -> None:
        """Kill an in-flight engine process without waiting on its lock."""
        if self._phase != "thinking":
            return
        self._generation += 1
        teacher = self._active_teacher
        if teacher is not None:
            try:
                teacher.close()
            except Exception:  # noqa: BLE001 - stop must never raise
                pass
        self._active_teacher = None
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
            if kind in ("train_progress", "train_done", "train_error"):
                if generation != self._train_generation:
                    continue
                if kind == "train_progress":
                    self._show_training_progress(payload)
                    continue
                self._phase = "idle"
                self._train_thread = None
                self._train_stop = None
                self.btn_train_start.setEnabled(True)
                self.btn_train_stop.setEnabled(False)
                if kind == "train_done":
                    self.status_label.setText("训练完成")
                    self._show_training_result(payload)
                else:
                    self._show_error("训练失败：" + str(payload))
                continue
            if kind in ("collect_progress", "collect_done", "collect_error"):
                if generation != self._collect_generation:
                    continue
                if kind == "collect_progress":
                    self._show_collection_update(payload)
                    continue
                self._phase = "idle"
                self._collect_thread = None
                self._collect_stop = None
                self.btn_collect_start.setEnabled(True)
                self.btn_collect_stop.setEnabled(False)
                if kind == "collect_done":
                    self.status_label.setText("采集完成")
                    self._show_collection_update(payload, done=True)
                else:
                    self._show_error("采集失败：" + str(payload))
                continue
            if generation != self._generation:
                continue  # a stop/new-game interrupted this thinking step
            if kind == "think":
                self._apply_engine_move(payload)
            elif kind == "vcf":
                self._phase = "idle"
                self._show_vcf(payload.get("vcf"))
                self.status_label.setText("算杀检查完成")
                self._refresh()
                self._maybe_engine_move()
            elif kind == "teacher":
                self._phase = "idle"
                self._show_teacher_report(payload)
                self.status_label.setText("教师打标完成")
                self._refresh()
                self._maybe_engine_move()
            else:
                self._phase = "idle"
                self._show_error(payload)

    def _check_vcf(self) -> None:
        """Ask the engine's upstream VCF solver about the current position."""
        if self._state is None or self._phase == "thinking":
            return
        if self._state.get("result") != "ongoing":
            self._show_error("对局已结束，无法进行算杀检查")
            return
        self._phase = "thinking"
        self.status_label.setText("算杀检查中…（可点「停止对局」中断）")
        self.error_label.setText("")
        client = self._client
        generation = self._generation
        max_nodes = self.spin_vcf.value()

        def worker() -> None:
            try:
                response = client.vcf(max_nodes=max_nodes)
                self._queue.put(("vcf", generation, response))
            except ExecutorError as exc:
                self._queue.put(("error", generation, str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _show_vcf(self, report) -> None:
        if not isinstance(report, dict):
            self.vcf_label.setText("算杀：未运行")
            return
        self.vcf_label.setText("算杀：" + vcf_tactics.describe(report))

    def _pool_status_text(self) -> str:
        path = ROOT / "data" / "pool" / "pool.sqlite3"
        if not path.exists():
            return "样本池：未开始"
        try:
            with SamplePool(path, create=False) as pool:
                stats = pool.stats()
        except PoolError as exc:
            return "样本池：不可读 — " + str(exc)
        return (f"样本池：{stats['pool_samples']}/{stats['capacity']}"
                f"（训练 {stats['train_samples']}，验证 {stats['validation_samples']}，"
                f"淘汰局 {stats['evicted_games']}）")

    def _start_collection(self) -> None:
        if self._phase in ("collecting", "thinking", "training"):
            return
        try:
            identity = rapfi_teacher.load_lock()
            install = rapfi_teacher.check_install(identity)
        except rapfi_teacher.TeacherError as exc:
            self._show_error("教师不可用：" + str(exc))
            return
        if not install.ok:
            self._show_error("教师不可用：" + "；".join(install.problems))
            return
        games = self.spin_collect_games.value()
        capacity = self.spin_collect_capacity.value()
        budget = self.spin_collect_budget.value()
        config = collect.CollectConfig(
            games=games, depth=2, max_plies=24, opening_plies=2,
            opening_region=7, sample_every=3, budget_ms=budget,
            vcf_nodes=5000, vcf_plies=4, seed=int(time.time()) % 100000,
            model_id="handcrafted")
        stop_event = threading.Event()
        self._collect_stop = stop_event
        self._collect_generation += 1
        generation = self._collect_generation
        self._phase = "collecting"
        self.status_label.setText("采集运行中…")
        self.btn_collect_start.setEnabled(False)
        self.btn_collect_stop.setEnabled(True)
        exe = self._exe
        pool_path = ROOT / "data" / "pool" / "pool.sqlite3"

        def worker() -> None:
            pool = None
            try:
                pool = SamplePool(pool_path, capacity=capacity)
                pool.set_config("validation_games", 1)
                with ExecutorClient(exe) as game_client, \
                        ExecutorClient(exe) as label_client, \
                        rapfi_teacher.RapfiTeacher(identity=identity) as teacher:
                    summary = collect.collect(
                        pool, config, game_client, label_client, teacher,
                        stop_event=stop_event,
                        progress=lambda update: self._queue.put(
                            ("collect_progress", generation, update)))
                self._queue.put(("collect_done", generation, summary))
            except Exception as exc:  # noqa: BLE001 - surfaced in the UI
                self._queue.put(("collect_error", generation, str(exc)))
            finally:
                if pool is not None:
                    pool.close()

        self._collect_thread = threading.Thread(target=worker, daemon=True)
        self._collect_thread.start()

    def _stop_collection(self) -> None:
        if self._collect_stop is None:
            return
        self._collect_stop.set()
        self.btn_collect_stop.setEnabled(False)
        self.status_label.setText("停止采集中…")

    def _show_collection_update(self, update: dict, *, done: bool = False) -> None:
        parts = [
            f"生成 {update.get('generated', 0)}",
            f"有效 {update.get('valid', 0)}",
            f"入池 {update.get('inserted', 0)}",
            f"池 {update.get('pool_samples', 0)}/{update.get('capacity', 0)}",
        ]
        rate = update.get("effective_per_second")
        if rate:
            parts.append(f"有效/秒 {rate:.2f}")
        if update.get("validation_samples"):
            parts.append(f"验证 {update['validation_samples']}")
        if update.get("evicted_games"):
            parts.append(f"淘汰局 {update['evicted_games']}")
        if update.get("staged_games"):
            parts.append(f"暂存 {update['staged_games']}")
        if done:
            parts.append("已完成" if not update.get("stopped") else "已停止")
        self.collect_label.setText("采集：" + "，".join(parts))
        filters = update.get("filters") or {}
        if filters:
            top = ", ".join(f"{name}={count}"
                            for name, count in sorted(filters.items()))
            self.collect_label.setText(self.collect_label.text() + "；过滤 " + top)
        if done:
            self.collect_pool_label.setText(self._pool_status_text())

    def _start_training(self) -> None:
        if self._phase != "idle":
            return
        pool_path = self.edit_pool.text().strip()
        output_dir = self.edit_train_output.text().strip()
        if not pool_path or not output_dir:
            self._show_error("需要样本池路径和输出目录")
            return
        steps = self.spin_train_steps.value()
        batch = self.spin_train_batch.value()
        lr = self.spin_train_lr.value()
        loss = "mse" if self.combo_train_loss.currentIndex() == 0 else "smooth_l1"
        stop_event = threading.Event()
        self._train_stop = stop_event
        self._train_generation += 1
        generation = self._train_generation
        self._phase = "training"
        self.status_label.setText("训练运行中…（可点「停止训练」保存当前状态）")
        self.btn_train_start.setEnabled(False)
        self.btn_train_stop.setEnabled(True)
        self.train_label.setText("训练：正在加载数据与初始化…")
        self.train_result.setText("")

        def worker() -> None:
            try:
                import value_train
                config = value_train.TrainConfig(
                    pool=pool_path, output_dir=output_dir,
                    seed=int(time.time()) % 100000, steps=steps,
                    batch_size=batch, lr=lr, loss=loss, device="auto",
                    val_every=max(5, steps // 5),
                    checkpoint_every=max(10, steps // 2),
                    log_every=max(1, steps // 10))
                summary = value_train.train_value(
                    config, stop_event=stop_event,
                    progress=lambda update: self._queue.put(
                        ("train_progress", generation, update)))
                self._queue.put(("train_done", generation, summary))
            except Exception as exc:  # noqa: BLE001 - surfaced in the UI
                self._queue.put(("train_error", generation, str(exc)))

        self._train_thread = threading.Thread(target=worker, daemon=True)
        self._train_thread.start()

    def _stop_training(self) -> None:
        if self._train_stop is None:
            return
        self._train_stop.set()
        self.btn_train_stop.setEnabled(False)
        self.status_label.setText("停止训练中（保存 checkpoint 与导出）…")

    def _show_training_progress(self, update: dict) -> None:
        parts = [f"step {update.get('step', '?')}"]
        if update.get("train_loss") is not None:
            parts.append(f"train {update['train_loss']:.5f}")
        if update.get("val_loss") is not None:
            parts.append(f"val {update['val_loss']:.5f}")
        if update.get("val_mae") is not None:
            parts.append(f"val MAE {update['val_mae']:.4f}")
        if update.get("elapsed_s") is not None:
            parts.append(f"{update['elapsed_s']:.1f}s")
        self.train_label.setText("训练：" + "，".join(parts))

    def _show_training_result(self, summary: dict) -> None:
        export = summary.get("export") or {}
        passed = export.get("passed")
        validation = summary.get("validation") or {}
        lines = [
            f"训练：{summary.get('steps', 0)} 步完成"
            + ("（已停止）" if summary.get("stopped") else ""),
            f"val loss {validation.get('loss')}, val MAE {validation.get('mae')}",
        ]
        if export:
            model = export.get("model") or {}
            lines.append("导出保真度：" + ("通过" if passed else "未通过"))
            if model.get("path"):
                lines.append("模型 " + str(model["path"]))
            if export.get("report"):
                lines.append("报告 " + str(export["report"]))
        self.train_result.setText("；".join(lines))

    def _teacher_status_text(self) -> str:
        try:
            identity = rapfi_teacher.load_lock()
            install = rapfi_teacher.check_install(identity)
        except rapfi_teacher.TeacherError as exc:
            return "教师：不可用 — " + str(exc)
        if not install.ok:
            return "教师：不可用 — " + "；".join(install.problems)
        return ("教师：" + identity.describe()
                + "；未安装时不产生任何标签")

    def _label_teacher(self) -> None:
        """Labels the current position with the local teacher and saves it."""
        if self._state is None or self._phase == "thinking":
            return
        try:
            identity = rapfi_teacher.load_lock()
            install = rapfi_teacher.check_install(identity)
        except rapfi_teacher.TeacherError as exc:
            self._show_error("教师不可用：" + str(exc))
            return
        if not install.ok:
            self._show_error("教师不可用：" + "；".join(install.problems))
            return
        self._phase = "thinking"
        self.status_label.setText("教师打标中…（可点「停止对局」中断）")
        self.error_label.setText("")
        generation = self._generation
        state = self._state
        cells = state["cells"]
        side = state["side_to_move"]
        history = state.get("history", [])
        ply = len(history)
        last_move = None
        if history:
            last_move = history[-1][1] * SIZE + history[-1][0]
        game_id = self._game_id
        budget_ms = self.spin_teacher_budget.value()
        store_path = ROOT / "data" / "teacher" / "samples.sqlite3"
        exe = self._exe

        def worker() -> None:
            teacher = None
            try:
                teacher = rapfi_teacher.RapfiTeacher(identity=identity)
                self._active_teacher = teacher
                with ExecutorClient(exe) as executor, teacher, \
                        TeacherStore(store_path) as store:
                    report = teacher_label.label_position(
                        executor=executor, teacher=teacher, store=store,
                        cells=cells, side=side, source_kind="game",
                        source_id=game_id, ply=ply, last_move=last_move,
                        budget_ms=budget_ms)
                self._queue.put(("teacher", generation, report))
            except Exception as exc:  # noqa: BLE001 - surfaced in the UI
                self._queue.put(("error", generation, str(exc)))
            finally:
                self._active_teacher = None

        threading.Thread(target=worker, daemon=True).start()

    def _show_teacher_report(self, report: dict) -> None:
        if not isinstance(report, dict):
            self.teacher_result.setText("样本：未打标")
            return
        if report.get("accepted"):
            value = report.get("value")
            text = (f"已保存：raw {report.get('raw_eval')} → value "
                    f"{value:+.3f}" if value is not None else "已保存")
        else:
            text = "未入库：" + str(report.get("filter_reason"))
        counts = report.get("counts") or {}
        detail = (f"；教师状态 {report.get('teacher_status')}"
                  f"；预算 {report.get('budget_ms')} ms"
                  f"；有效 {counts.get('accepted', 0)} / "
                  f"请求 {counts.get('total', 0)}"
                  f"（未知 {counts.get('unknown', 0)}，"
                  f"VCF 未知 {counts.get('vcf_unknown', 0)}）")
        extra = ""
        if report.get("suggested_move"):
            extra = f"；建议着法 {report['suggested_move']}（与评分分开记录）"
        message = report.get("message")
        if message and not report.get("accepted"):
            extra += "；" + str(message)
        self.teacher_result.setText("样本：" + text + detail + extra)

    def _apply_engine_move(self, response: dict) -> None:
        if not response.get("ok"):
            self._phase = "idle"
            self._show_error("引擎选点失败：" + str(response.get("error", response)))
            return
        self._show_vcf(response.get("vcf"))
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
        if self._collect_stop is not None:
            self._collect_stop.set()
        if self._collect_thread is not None:
            self._collect_thread.join(timeout=10)
        if self._train_stop is not None:
            self._train_stop.set()
        if self._train_thread is not None:
            self._train_thread.join(timeout=20)
        teacher = self._active_teacher
        if teacher is not None:
            try:
                teacher.close()
            except Exception:  # noqa: BLE001 - shutdown must never raise
                pass
        self._active_teacher = None
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
