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

from PySide6.QtCore import QPointF, QRect, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
    QGraphicsDropShadowEffect, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QMainWindow, QMessageBox, QPushButton, QRadioButton,
    QScrollArea, QSizePolicy, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
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
from model_registry import ModelRegistry, DEFAULT_REGISTRY  # noqa: E402
from run_support import file_identity  # noqa: E402
from history_panel import HistoryPanel  # noqa: E402
from training_task_panel import TrainingTaskPanel  # noqa: E402
from ui_components import (  # noqa: E402
    CollapsibleSection, SegmentedControl, apply_theme, mark_button, page_title,
)
from settings_sheet import SettingsSheet  # noqa: E402

SIZE = 15
BOARD_MIN_SIDE = 260

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
NO_MODEL_TEXT = "手工威胁评估（无模型）"


def discover_models() -> list[Path]:
    """Local GMVALUE1 candidates found under weights/ for the model picker."""
    root = ROOT / "weights"
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.gmvalue"))


def model_args(model: str | None) -> list[str]:
    return ["--value-model", str(model)] if model else []


def short_identity(text: str, keep: int = 8) -> str:
    """Shorten a long ``name@40-char-commit`` token so labels can wrap."""
    head, sep, rest = text.partition("@")
    if not sep:
        return text
    parts = rest.split(" ", 1)
    commit = parts[0]
    if len(commit) <= keep:
        return text
    tail = " " + parts[1] if len(parts) > 1 else ""
    return f"{head}@{commit[:keep]}…{tail}"


def start_executor(exe: str, model: str | None) -> tuple[ExecutorClient, dict]:
    """Start the engine and return (client, about).

    Ticket 07: an explicitly selected model must either load or fail loudly.
    The engine exits with a diagnostic on stderr; surface that text instead of
    silently pretending the model is in use.
    """
    client = ExecutorClient(exe, *model_args(model))
    try:
        about = client.about()
    except ExecutorError as exc:
        # Let the stderr drain thread collect the engine's explanation.
        deadline = time.monotonic() + 2.0
        diagnostics = ""
        while time.monotonic() < deadline:
            diagnostics = client.diagnostics().strip()
            if diagnostics:
                break
            time.sleep(0.05)
        client.close()
        detail = diagnostics or str(exc)
        raise ExecutorError(f"模型加载失败：{detail}") from exc
    if not about.get("ok", True):
        client.close()
        raise ExecutorError(f"模型加载失败：{about}")
    return client, about


class BoardWidget(QWidget):
    moveClicked = Signal(int, int)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(BOARD_MIN_SIDE, BOARD_MIN_SIDE)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.PointingHandCursor)
        self.cells: list[int] = [0] * (SIZE * SIZE)
        self.legal: set[tuple[int, int]] = set()
        self.forbidden: dict[tuple[int, int], str] = {}
        self.last_move: tuple[int, int] | None = None
        self.markers = True
        self.preview_mode = False
        self.banner = ""
        self.banner_sub = ""
        self.dim = False

    def set_position(self, cells, legal, forbidden, last_move, markers) -> None:
        self.cells = cells
        self.legal = legal
        self.forbidden = forbidden
        self.last_move = last_move
        self.markers = markers
        self.preview_mode = False
        self.banner = ""
        self.banner_sub = ""
        self.dim = False
        self.update()

    def set_preview(self, cells, last_move, banner: str, subtitle: str = "",
                    dim: bool = False) -> None:
        """Read-only mirror; legal/forbidden overlays are deliberately empty."""
        self.cells = cells
        self.legal = set()
        self.forbidden = {}
        self.last_move = last_move
        self.markers = False
        self.preview_mode = True
        self.banner = banner
        self.banner_sub = subtitle
        self.dim = dim
        self.update()

    def _metrics(self) -> tuple[float, float, float]:
        """Cell spacing and grid origin for the current widget size.

        The grid always stays square and centred while the board surface
        fills the card, so dragging the window resizes it instead of
        stretching or clipping it.
        """
        side = min(self.width(), self.height())
        margin = max(10.0, side * 0.05)
        cell = (side - 2 * margin) / (SIZE - 1)
        return (cell,
                (self.width() - side) / 2 + margin,
                (self.height() - side) / 2 + margin)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(233, 224, 207))

        cell, ox, oy = self._metrics()
        end = (SIZE - 1) * cell
        painter.setPen(QPen(QColor(190, 176, 153), max(1.0, cell / 30)))
        for i in range(SIZE):
            offset = i * cell
            painter.drawLine(QPointF(ox, oy + offset),
                             QPointF(ox + end, oy + offset))
            painter.drawLine(QPointF(ox + offset, oy),
                             QPointF(ox + offset, oy + end))

        dot = max(2.0, cell * 0.09)
        painter.setBrush(QColor(150, 136, 114))
        painter.setPen(QPen(Qt.NoPen))
        for sx, sy in ((3, 3), (11, 3), (3, 11), (11, 11), (7, 7)):
            painter.drawEllipse(QPointF(ox + sx * cell, oy + sy * cell),
                                dot, dot)

        if self.markers:
            painter.setPen(QPen(Qt.NoPen))
            painter.setBrush(QColor(0, 122, 255, 72))
            marker = max(1.5, cell * 0.11)
            for x, y in self.legal:
                painter.drawEllipse(QPointF(ox + x * cell, oy + y * cell),
                                    marker, marker)
            painter.setPen(QPen(QColor(255, 59, 48, 190),
                                max(1.0, cell * 0.07)))
            cross = max(2.0, cell * 0.16)
            for x, y in self.forbidden:
                cx, cy = ox + x * cell, oy + y * cell
                painter.drawLine(QPointF(cx - cross, cy - cross),
                                 QPointF(cx + cross, cy + cross))
                painter.drawLine(QPointF(cx - cross, cy + cross),
                                 QPointF(cx + cross, cy - cross))

        radius = cell * 0.42
        outline = QPen(QColor(60, 60, 67, 70), max(1.0, cell * 0.04))
        for m, c in enumerate(self.cells):
            if c == 0:
                continue
            x, y = m % SIZE, m // SIZE
            cx, cy = ox + x * cell, oy + y * cell
            # A soft, low-contrast shadow keeps the stones feeling placed on
            # the board instead of drawn on top of it.
            painter.setPen(QPen(Qt.NoPen))
            painter.setBrush(QColor(0, 0, 0, 34))
            painter.drawEllipse(QPointF(cx + radius * 0.08,
                                        cy + radius * 0.12),
                                radius, radius)
            painter.setBrush(QColor(28, 28, 30) if c == 1
                             else QColor(255, 255, 255))
            painter.setPen(outline)
            painter.drawEllipse(QPointF(cx, cy), radius, radius)

        if self.last_move is not None:
            x, y = self.last_move
            ring = radius + max(2.0, cell * 0.1)
            painter.setPen(QPen(QColor(0, 122, 255), max(1.5, cell * 0.08)))
            painter.setBrush(QBrush())
            painter.drawEllipse(QPointF(ox + x * cell, oy + y * cell),
                                ring, ring)

        if self.preview_mode and self.banner:
            if self.dim:
                painter.setPen(QPen(Qt.NoPen))
                painter.setBrush(QColor(245, 245, 247, 138))
                painter.drawRect(self.rect())
            font = QFont(self.font())
            font.setPointSize(10)
            font.setBold(True)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            banner_width = max(96, metrics.horizontalAdvance(self.banner) + 48)
            if self.banner_sub:
                sub_font = QFont(self.font())
                sub_font.setPointSize(8)
                painter.setFont(sub_font)
                sub_metrics = painter.fontMetrics()
                banner_width = max(
                    banner_width,
                    sub_metrics.horizontalAdvance(self.banner_sub) + 48)
            banner_height = 44 if self.banner_sub else 30
            rect = QRect(16, 14,
                         min(banner_width, self.rect().width() - 32),
                         banner_height)
            painter.setPen(QPen(Qt.NoPen))
            painter.setBrush(QColor(29, 29, 31, 218))
            painter.drawRoundedRect(rect, 12, 12)
            painter.setFont(font)
            painter.setPen(QColor(255, 255, 255))
            text_rect = rect.adjusted(12, 4, -10, -4)
            if self.banner_sub:
                painter.drawText(text_rect.adjusted(0, 2, 0, -16),
                                 Qt.AlignLeft | Qt.AlignVCenter, self.banner)
                painter.setFont(QFont(self.font()))
                painter.setPen(QColor(235, 235, 240))
                painter.drawText(text_rect.adjusted(0, 20, 0, 0),
                                 Qt.AlignLeft | Qt.AlignVCenter, self.banner_sub)
            else:
                painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter,
                                 self.banner)
        painter.end()

    def mousePressEvent(self, event) -> None:
        if self.preview_mode:
            return
        if event.button() != Qt.LeftButton:
            return
        pos = event.position()
        cell, ox, oy = self._metrics()
        x = round((pos.x() - ox) / cell)
        y = round((pos.y() - oy) / cell)
        if 0 <= x < SIZE and 0 <= y < SIZE:
            if ((pos.x() - (ox + x * cell)) ** 2
                    + (pos.y() - (oy + y * cell)) ** 2
                    <= (cell * 0.45) ** 2):
                self.moveClicked.emit(x, y)


class MainWindow(QMainWindow):
    def __init__(self, exe: str, think_ms: int, model: str | None = None,
                 registry: str | Path = DEFAULT_REGISTRY,
                 evaluations: str | Path = ROOT / "data" / "evaluations",
                 pool: str | Path = ROOT / "data" / "pool" / "pool.sqlite3",
                 collection_seed: int | None = None,
                 teacher_store: str | Path = ROOT / "data/teacher/samples.sqlite3",
                 restore_geometry: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("GomokuNNUE · 有禁手五子棋")
        self._restore_geometry = restore_geometry
        self._settings = QSettings("GomokuNNUE", "DesktopClient")
        app = QApplication.instance()
        if app is not None:
            apply_theme(app)
        self._exe = exe
        self._think_ms = think_ms
        self._registry = ModelRegistry(registry)
        self._evaluations = evaluations
        self._pool_path = Path(pool)
        self._collection_seed = collection_seed
        self._teacher_store_path = Path(teacher_store)
        self._model = str(model) if model else None
        self._startup_error = ""
        if model is None:
            try:
                current = self._registry.resolve()
                if current:
                    self._model = current["path"]
            except (ValueError, OSError) as exc:
                self._startup_error = "正式版本不可用：" + str(exc)
        try:
            self._client, self._about = start_executor(exe, self._model)
        except ExecutorError as exc:
            # Keep the client usable, but never pretend the rejected model is
            # loaded: the identity panel and error label say what happened.
            self._startup_error = str(exc)
            self._model = None
            self._client, self._about = start_executor(exe, None)
        self._state: dict | None = None
        self._phase = "idle"  # idle | thinking | game_over
        self._player_color = "black"
        self._game_id = uuid.uuid4().hex[:12]
        self._active_teacher = None
        self._collect_stop = None
        self._collect_pause = None
        self._collect_thread = None
        self._collect_generation = 0
        self._train_stop = None
        self._train_pause = None
        self._train_resume = None
        self._train_target = 0
        self._train_thread = None
        self._train_generation = 0
        self._generation = 0
        self._preview: dict | None = None
        self._preview_key: tuple | None = None
        self._preview_last_at = 0.0
        self._live_forced = False
        self._background_active = False
        self._recoverable_count = 0
        self._training_task_panel_busy = False
        self._preview_paused = False
        self._preview_dim = False
        self._preview_training = None
        self._live_hold_token = 0
        self._queue: queue.Queue = queue.Queue()
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_queue)
        self._timer.start(40)
        self._new_game()
        if self._startup_error:
            self._show_error(self._startup_error)

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("Root")
        outer = QVBoxLayout(central)
        outer.setContentsMargins(20, 14, 20, 16)
        outer.setSpacing(8)

        header = QWidget()
        header_row = QHBoxLayout(header)
        header_row.setContentsMargins(2, 0, 2, 0)
        header_row.setSpacing(10)
        header_row.addWidget(page_title("五子棋", "本地对弈 · 训练 · 测评"))
        header_row.addStretch(1)
        self.combo_model = QComboBox()
        self.combo_model.setObjectName("ModelCapsule")
        self.combo_model.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_model.setMinimumContentsLength(14)
        self.combo_model.setMinimumWidth(170)
        self.combo_model.setMaximumWidth(280)
        self._reload_model_choices()
        self.combo_model.activated.connect(self._model_capsule_activated)
        header_row.addWidget(self.combo_model)
        self.btn_settings = QPushButton("设置")
        mark_button(self.btn_settings, "quiet")
        header_row.addWidget(self.btn_settings)
        self.header_status = QLabel("准备就绪")
        self.header_status.setObjectName("Pill")
        header_row.addWidget(self.header_status)
        outer.addWidget(header)

        activity_row = QHBoxLayout()
        activity_row.setContentsMargins(2, 0, 2, 0)
        activity_row.setSpacing(8)
        self.activity_label = QLabel("准备就绪")
        self.activity_label.setObjectName("Caption")
        activity_row.addWidget(self.activity_label)
        activity_row.addStretch(1)
        self.btn_live = QPushButton("LIVE")
        self.btn_live.setObjectName("Live")
        self.btn_live.setCursor(Qt.PointingHandCursor)
        self.btn_live.hide()
        self.btn_live.clicked.connect(self._toggle_live)
        activity_row.addWidget(self.btn_live)
        outer.addLayout(activity_row)

        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #FF3B30;")
        self.error_label.hide()
        outer.addWidget(self.error_label)

        content = QWidget()
        root = QHBoxLayout(content)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(18)

        self.board = BoardWidget()
        self.board.moveClicked.connect(self._on_player_move)
        board_card = QFrame()
        board_card.setObjectName("BoardCard")
        board_card.setSizePolicy(QSizePolicy.Policy.Expanding,
                                 QSizePolicy.Policy.Expanding)
        board_layout = QVBoxLayout(board_card)
        board_layout.setContentsMargins(12, 12, 12, 12)
        board_layout.addWidget(self.board, 1)
        shadow = QGraphicsDropShadowEffect(board_card)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 26))
        board_card.setGraphicsEffect(shadow)
        root.addWidget(board_card, 1)

        panel = QVBoxLayout()
        panel.setSpacing(12)
        tabs = QTabWidget()
        self.tabs = tabs
        play_page = QWidget()
        play_layout = QVBoxLayout(play_page)
        play_layout.setContentsMargins(0, 0, 0, 0)
        self.play_live_hint = QLabel("后台任务运行中 · 棋盘为只读实时镜像")
        self.play_live_hint.setObjectName("Caption")
        self.play_live_hint.setWordWrap(True)
        self.play_live_hint.hide()
        play_layout.addWidget(self.play_live_hint)
        train_page = QWidget()
        train_layout = QVBoxLayout(train_page)
        train_layout.setContentsMargins(0, 0, 0, 0)
        eval_page = QWidget()
        eval_layout = QVBoxLayout(eval_page)
        eval_layout.setContentsMargins(0, 0, 0, 0)

        # ---- 对弈 -----------------------------------------------------
        setup = QGroupBox("对局")
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
        mark_button(self.btn_new, "primary")
        self.btn_stop = QPushButton("停止对局")
        self.btn_stop.clicked.connect(self._stop_game)
        mark_button(self.btn_stop, "danger")
        btns.addWidget(self.btn_new)
        btns.addWidget(self.btn_stop)
        sv.addLayout(btns)
        play_layout.addWidget(setup)

        advanced_play_layout = QVBoxLayout()
        advanced_play_layout.setContentsMargins(0, 0, 0, 0)
        advanced_play_layout.setSpacing(8)
        self.chk_markers = QCheckBox("显示合法落点 / 禁手原因")
        self.chk_markers.setChecked(True)
        self.chk_markers.toggled.connect(self._on_markers_toggled)
        advanced_play_layout.addWidget(self.chk_markers)
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
        advanced_play_layout.addLayout(vcf_row)
        think = QHBoxLayout()
        think.addWidget(QLabel("引擎思考(ms)"))
        self.spin_think = QSpinBox()
        self.spin_think.setRange(100, 30000)
        self.spin_think.setValue(self._think_ms)
        think.addWidget(self.spin_think)
        advanced_play_layout.addLayout(think)
        self.play_advanced = CollapsibleSection("高级对局设置",
                                                advanced_play_layout)
        self.play_advanced.embed()

        # Kept for tests/backward compatibility; shown in the settings sheet.
        self.turn_label = QLabel("行棋方：-")
        self.status_label = QLabel("")
        self.result_label = QLabel("对局进行中")
        self.vcf_label = QLabel("算杀：未检查")
        self.vcf_label.setWordWrap(True)
        engine_identity = str(self._about.get("vcf_engine", "未加载"))
        self.vcf_engine_label = QLabel(
            "算杀引擎：" + short_identity(engine_identity))
        self.vcf_engine_label.setWordWrap(True)
        self.vcf_engine_label.setToolTip(engine_identity)
        self.replay_label = QLabel("")
        self.replay_label.setWordWrap(True)
        details_layout = QVBoxLayout()
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(4)
        for widget in (self.turn_label, self.status_label, self.result_label,
                       self.vcf_engine_label, self.vcf_label,
                       self.replay_label):
            widget.setObjectName("Caption")
            details_layout.addWidget(widget)
        self.play_details = CollapsibleSection("对局状态与详情", details_layout)
        self.play_details.embed()

        record_layout = QVBoxLayout()
        record_layout.setContentsMargins(0, 0, 0, 0)
        self.moves_list = QListWidget()
        self.moves_list.setMaximumHeight(180)
        record_layout.addWidget(self.moves_list)
        self.record_section = CollapsibleSection("着法记录", record_layout)
        play_layout.addWidget(self.record_section)
        play_layout.addStretch(1)

        # ---- 训练 -----------------------------------------------------
        self.training_mode = SegmentedControl([
            ("collect", "采集数据"), ("fromzero", "从0训练"),
            ("selfplay", "自我训练")], "collect")
        train_layout.addWidget(self.training_mode)

        collection = QGroupBox("运行")
        cv = QVBoxLayout(collection)
        self.collect_pool_label = QLabel(self._pool_status_text())
        self.collect_pool_label.setWordWrap(True)
        self.collect_pool_label.setObjectName("Caption")
        cv.addWidget(self.collect_pool_label)
        collect_buttons = QHBoxLayout()
        self.btn_collect_start = QPushButton("启动采集")
        self.btn_collect_start.clicked.connect(self._start_collection)
        mark_button(self.btn_collect_start, "primary")
        self.btn_collect_pause = QPushButton("暂停采集")
        self.btn_collect_pause.clicked.connect(self._pause_collection)
        self.btn_collect_pause.setEnabled(False)
        self.btn_collect_stop = QPushButton("停止采集")
        self.btn_collect_stop.clicked.connect(self._stop_collection)
        self.btn_collect_stop.setEnabled(False)
        mark_button(self.btn_collect_stop, "danger")
        collect_buttons.addWidget(self.btn_collect_start)
        collect_buttons.addWidget(self.btn_collect_pause)
        collect_buttons.addWidget(self.btn_collect_stop)
        cv.addLayout(collect_buttons)
        self.collect_label = QLabel("采集：未开始")
        self.collect_label.setWordWrap(True)
        self.collect_label.setObjectName("Caption")
        cv.addWidget(self.collect_label)
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
        collect_row.addStretch(1)
        collect_advanced_layout = QVBoxLayout()
        collect_advanced_layout.setContentsMargins(0, 0, 0, 0)
        collect_advanced_layout.addLayout(collect_row)
        collect_row = QHBoxLayout()
        collect_row.addWidget(QLabel("预算(ms)"))
        self.spin_collect_budget = QSpinBox()
        self.spin_collect_budget.setRange(100, 2000)
        self.spin_collect_budget.setSingleStep(100)
        self.spin_collect_budget.setValue(300)
        collect_row.addWidget(self.spin_collect_budget)
        collect_row.addStretch(1)
        collect_advanced_layout.addLayout(collect_row)
        self.collect_advanced = CollapsibleSection("采集参数",
                                                   collect_advanced_layout)
        self.collect_advanced.embed()
        self.collection_card = collection
        train_layout.addWidget(collection)

        training = QGroupBox("运行")
        tv2 = QVBoxLayout(training)
        train_buttons = QHBoxLayout()
        self.btn_train_start = QPushButton("开始训练")
        self.btn_train_start.clicked.connect(self._start_training)
        self.btn_train_pause = QPushButton("暂停训练")
        self.btn_train_pause.clicked.connect(self._pause_training)
        self.btn_train_pause.setEnabled(False)
        self.btn_train_resume = QPushButton("继续上次训练")
        self.btn_train_resume.clicked.connect(self._resume_training)
        self.btn_train_resume.setEnabled(False)
        self.btn_train_stop = QPushButton("停止训练")
        self.btn_train_stop.clicked.connect(self._stop_training)
        self.btn_train_stop.setEnabled(False)
        mark_button(self.btn_train_stop, "danger")
        for button in (self.btn_train_start, self.btn_train_pause,
                       self.btn_train_resume, self.btn_train_stop):
            train_buttons.addWidget(button)
        tv2.addLayout(train_buttons)
        self.train_label = QLabel("训练：未开始")
        self.train_label.setWordWrap(True)
        self.train_label.setObjectName("Caption")
        tv2.addWidget(self.train_label)
        self.train_result = QLabel("")
        self.train_result.setWordWrap(True)
        self.train_result.setObjectName("Caption")
        tv2.addWidget(self.train_result)

        train_advanced_layout = QVBoxLayout()
        train_advanced_layout.setContentsMargins(0, 0, 0, 0)
        train_advanced_layout.setSpacing(6)
        pool_row = QHBoxLayout()
        pool_row.addWidget(QLabel("样本池"))
        self.edit_pool = QLineEdit(str(self._pool_path))
        pool_row.addWidget(self.edit_pool)
        train_advanced_layout.addLayout(pool_row)
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("输出"))
        self.edit_train_output = QLineEdit(str(ROOT / "weights" / "value-run"))
        out_row.addWidget(self.edit_train_output)
        train_advanced_layout.addLayout(out_row)
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
        train_row.addStretch(1)
        train_advanced_layout.addLayout(train_row)
        train_row = QHBoxLayout()
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
        train_row.addStretch(1)
        train_advanced_layout.addLayout(train_row)
        self.train_advanced = CollapsibleSection("训练参数",
                                                 train_advanced_layout)
        self.train_advanced.embed()
        self.training_card = training
        train_layout.addWidget(training)

        self.training_task_panel = TrainingTaskPanel(
            self._exe, self._pool_path, self._registry.root, self._evaluations,
            ROOT / "weights/teacher-tasks",
            lambda: self._phase in ("idle", "game_over"))
        self.training_task_panel.busyChanged.connect(self._training_task_busy)
        self.training_task_panel.published.connect(self._apply_model_path)
        self.training_task_panel.previewReceived.connect(self._on_preview)
        self.training_task_panel.stateChanged.connect(self._on_task_state)
        self.training_task_panel.recoverableChanged.connect(
            self._recoverable_changed)
        train_layout.addWidget(self.training_task_panel)
        self.btn_recovery_link = QPushButton("恢复未完成任务 →")
        mark_button(self.btn_recovery_link, "quiet")
        self.btn_recovery_link.hide()
        self.btn_recovery_link.clicked.connect(
            lambda: self.settings_sheet.show("recovery"))
        train_layout.addWidget(self.btn_recovery_link)
        self.training_task_panel._refresh_recoverable()
        train_layout.addStretch(1)
        self.training_mode.changed.connect(self._training_mode_changed)
        self._training_mode_changed("collect")

        # ---- 测评 -----------------------------------------------------
        self.history_panel = HistoryPanel(
            self._exe, self._registry.root, self._evaluations,
            lambda: self._phase in ("idle", "game_over"))
        self.history_panel.edit_candidate.setText(self._model or "")
        self.history_panel.busyChanged.connect(self._evaluation_busy)
        self.history_panel.published.connect(self._apply_model_path)
        self.history_panel.previewReceived.connect(self._on_preview)
        eval_layout.addWidget(self.history_panel)
        eval_layout.addStretch(1)

        tabs.addTab(play_page, "对弈")
        tabs.addTab(train_page, "训练")
        tabs.addTab(eval_page, "测评")
        panel.addWidget(tabs)

        panel_widget = QWidget()
        panel_widget.setLayout(panel)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setMinimumWidth(360)
        scroll.setWidget(panel_widget)
        root.addWidget(scroll, 1)
        outer.addWidget(content, 1)
        self.setCentralWidget(central)
        self._apply_window_size()

        # ---- model details + settings sheet ---------------------------
        teacher_layout = QVBoxLayout()
        teacher_layout.setContentsMargins(0, 0, 0, 0)
        teacher_layout.setSpacing(8)
        self.teacher_label = QLabel(self._teacher_status_text())
        self.teacher_label.setWordWrap(True)
        self.teacher_label.setObjectName("Caption")
        teacher_layout.addWidget(self.teacher_label)
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
        teacher_layout.addLayout(teacher_row)
        self.teacher_result = QLabel("样本：未打标")
        self.teacher_result.setWordWrap(True)
        self.teacher_result.setObjectName("Caption")
        teacher_layout.addWidget(self.teacher_result)
        self.teacher_section = CollapsibleSection("教师打标（Rapfi）",
                                                  teacher_layout)
        self.teacher_section.embed()

        self.eval_label = QLabel("")
        self.eval_label.setWordWrap(True)
        self.eval_label.setObjectName("Caption")
        self.btn_browse_model = QPushButton("浏览模型…")
        self.btn_browse_model.clicked.connect(self._browse_model)
        mark_button(self.btn_browse_model, "quiet")
        self.btn_apply_model = QPushButton("应用当前选择")
        self.btn_apply_model.clicked.connect(self._apply_model)
        mark_button(self.btn_apply_model, "quiet")
        self.btn_unload_model = QPushButton("手工评估")
        self.btn_unload_model.clicked.connect(
            lambda: self._apply_model_path(None))
        mark_button(self.btn_unload_model, "quiet")
        model_widget = QWidget()
        model_layout = QVBoxLayout(model_widget)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.addWidget(self.eval_label)
        row = QHBoxLayout()
        row.addWidget(self.btn_browse_model)
        row.addWidget(self.btn_apply_model)
        row.addWidget(self.btn_unload_model)
        model_layout.addLayout(row)
        self._update_eval_label()

        self.settings_sheet = SettingsSheet(self)
        self.settings_sheet.add_section("model", "模型", model_widget)
        self.settings_sheet.add_section("play", "高级对局设置",
                                        self.play_advanced)
        self.settings_sheet.add_section("play", "对局详情",
                                        self.play_details)
        self.settings_sheet.add_section("collect", "教师打标",
                                        self.teacher_section)
        # Teacher widgets are owned by MainWindow and deliberately not on the
        # main page. They are built just above the sheet to keep references.
        self.settings_sheet.add_section("collect", "采集参数",
                                        self.collect_advanced)
        self.settings_sheet.add_section("train", "从0训练参数",
                                        self.train_advanced)
        self.settings_sheet.add_section("tasks", "任务与资源",
                                        self.training_task_panel.settings)
        self.settings_sheet.add_section("recovery", "恢复未完成任务",
                                        self.training_task_panel.recovery_widget)
        self.settings_sheet.add_section("tasks", "任务指标",
                                        self.training_task_panel.metrics)
        self.settings_sheet.add_section("eval", "测评参数与对手",
                                        self.history_panel.advanced)
        self.settings_sheet.add_section("eval", "测评报告",
                                        self.history_panel.report)
        self.btn_settings.clicked.connect(lambda: self.settings_sheet.show())

    def _default_window_size(self) -> tuple[int, int]:
        """A small first launch; the window is freely resizable afterwards."""
        screen = self.screen()
        if screen is None:
            return 880, 520
        available = screen.availableGeometry()
        return (max(700, min(900, int(available.width() * 0.48))),
                max(420, min(560, int(available.height() * 0.48))))

    def _apply_window_size(self) -> None:
        self.setMinimumSize(700, 420)
        if self._restore_geometry:
            saved = self._settings.value("window/geometry")
            if saved is not None and self.restoreGeometry(saved):
                screen = self.screen()
                if screen is not None and screen.availableGeometry().intersects(
                        self.frameGeometry()):
                    return
        width, height = self._default_window_size()
        self.resize(width, height)

    def _training_mode_changed(self, key: str | None = None) -> None:
        key = key or self.training_mode.current()
        self.collection_card.setVisible(key == "collect")
        self.training_card.setVisible(key == "fromzero")
        self.training_task_panel.setVisible(key == "selfplay")

    def _recoverable_changed(self, count: int) -> None:
        if count:
            self.btn_recovery_link.setText(f"恢复 {count} 个未完成任务 →")
        self.btn_recovery_link.setVisible(bool(count))

    def _model_capsule_activated(self, index: int) -> None:
        if self._phase in ("thinking", "collecting", "training",
                           "evaluating", "continuous"):
            self._show_error("任务运行中，请先停止后再切换模型")
            self._reload_model_choices()
            return
        data = self.combo_model.itemData(index)
        if data == "__browse__":
            self._browse_model()
            self._reload_model_choices()
            return
        if self._phase in ("thinking", "collecting", "training",
                           "evaluating", "continuous"):
            self._show_error("任务运行中，请先停止后再切换模型")
            self._reload_model_choices()
            return
        self._apply_model_path(data if data else None)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "settings_sheet"):
            self.settings_sheet.reposition()

    def keyPressEvent(self, event) -> None:
        if (hasattr(self, "settings_sheet")
                and self.settings_sheet.visible()
                and event.key() == Qt.Key_Escape):
            self.settings_sheet.hide()
            return
        super().keyPressEvent(event)

    def _evaluation_busy(self, busy: bool) -> None:
        self._phase = "evaluating" if busy else "idle"
        self._set_background(busy, hold=not busy)
        if busy:
            self.activity_label.setText("测评运行中")
        elif not self._training_task_panel_busy:
            self.activity_label.setText("准备就绪")

    def _training_task_busy(self, busy: bool) -> None:
        self._phase = "continuous" if busy else "idle"
        self._training_task_panel_busy = busy
        self._set_background(busy, hold=not busy)
        if not busy:
            self.history_panel.refresh()
            self.activity_label.setText("准备就绪")
        else:
            self.activity_label.setText("自我训练运行中")

    def _set_background(self, active: bool, hold: bool = False) -> None:
        if active:
            self._live_hold_token += 1
            self._background_active = True
            self.play_live_hint.setText("后台任务运行中 · 棋盘为只读实时镜像")
            self.play_live_hint.show()
            self.btn_new.setEnabled(False)
            self._sync_live_button()
            return
        if hold and self._preview and self._live_showing():
            self._preview_paused = True
            self._preview_dim = True
            self._render_preview()
            self._live_hold_token += 1
            token = self._live_hold_token
            self.btn_live.hide()
            QTimer.singleShot(1200, lambda: self._finish_live_hold(token))
            return
        self._finish_live_hold(self._live_hold_token)

    def _finish_live_hold(self, token: int) -> None:
        if token != self._live_hold_token:
            return
        self._background_active = False
        self._preview = None
        self._preview_key = None
        self._live_forced = False
        self._preview_last_at = 0.0
        self.btn_live.hide()
        self.play_live_hint.hide()
        self.btn_new.setEnabled(True)
        if self._state is not None:
            self._refresh()

    def _player_midgame(self) -> bool:
        state = self._state
        return bool(state and state.get("result") == "ongoing"
                    and state.get("move_count", 0) > 0)

    def _live_showing(self) -> bool:
        return bool(self._preview and self._background_active
                    and (not self._player_midgame() or self._live_forced))

    def _sync_live_button(self) -> None:
        show = bool(self._preview and self._background_active
                    and self._player_midgame())
        self.btn_live.setVisible(show)
        if show:
            self.btn_live.setText("返回对局" if self._live_forced else "LIVE")
        if getattr(self, "play_live_hint", None) is not None:
            if self._live_showing():
                self.play_live_hint.setText(
                    "正在观看后台实时棋盘 · 只读；点「返回对局」继续")
            else:
                self.play_live_hint.setText("后台任务运行中 · 点 LIVE 查看")

    def _toggle_live(self) -> None:
        if not self._preview:
            return
        self._live_forced = not self._live_forced
        self._sync_live_button()
        if self._live_showing():
            self._render_preview()
        else:
            self._refresh()

    def _on_preview(self, payload: dict) -> None:
        cells = payload.get("cells")
        if not isinstance(cells, list) or len(cells) != SIZE * SIZE:
            return
        now = time.monotonic()
        key = (payload.get("kind"), payload.get("game"),
               payload.get("pair"))
        if self._preview_key is None or key != self._preview_key:
            same_kind = (self._preview_key is not None
                         and self._preview_key[0] == key[0])
            if same_kind and now - self._preview_last_at < 2.0:
                return
            self._preview_key = key
        self._preview = payload
        self._preview_last_at = now
        self._sync_live_button()
        if self._live_showing():
            self._live_forced = self._live_forced or not self._player_midgame()
            self._render_preview()

    def _render_preview(self) -> None:
        payload = self._preview
        if not payload:
            return
        kind = payload.get("kind")
        label = {"collect": "采集", "selfplay": "自我训练",
                 "evaluation": "测评"}.get(kind, "实时")
        number = payload.get("game") or payload.get("pair")
        title = f"LIVE · {label} {number}" if number else f"LIVE · {label}"
        side = "黑" if payload.get("side_to_move") == "black" else "白"
        subtitle = f"{side}方 · {payload.get('ply', 0)} 手"
        model_id = payload.get("model_id")
        if model_id:
            subtitle += f" · {model_id[:8]}"
        dim = bool(getattr(self, "_preview_dim", False))
        if getattr(self, "_preview_paused", False):
            title = "已暂停"
        elif getattr(self, "_preview_training", None):
            title = "训练中 · " + str(self._preview_training)
            dim = True
        last = payload.get("last_move")
        last_move = tuple(last) if isinstance(last, list) else None
        self.board.set_preview(payload["cells"], last_move, title, subtitle, dim)
        self.activity_label.setText(title + " · " + subtitle)

    def _on_task_state(self, state: dict) -> None:
        status = state.get("status")
        self._preview_paused = status == "paused"
        train = state.get("trainer") or {}
        data = state.get("data") or {}
        if train.get("phase") == "training":
            progress = train.get("progress") or {}
            self._preview_training = f"step {progress.get('step', '-')}"
            self.activity_label.setText(
                "自我训练 · 训练中 · step " + str(progress.get("step", "-")))
        else:
            self._preview_training = None
        if status == "paused" and self._preview:
            self._preview_dim = True
            self._render_preview()
        elif train.get("phase") == "training" and self._preview:
            self._preview_dim = True
            self._render_preview()
        elif data.get("phase") in ("collecting", "selfplay") and self._preview:
            self._preview_dim = False
            self._render_preview()
        elif status in ("stopped", "completed", "failed"):
            self._preview_training = None

    # ---- model selection ----------------------------------------------

    @staticmethod
    def _model_display(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(ROOT))
        except ValueError:
            return str(path)

    def _reload_model_choices(self) -> None:
        self.combo_model.clear()
        self.combo_model.addItem(NO_MODEL_TEXT, None)
        for path in discover_models():
            self.combo_model.addItem(self._model_display(path), str(path))
        index = self.combo_model.findData(self._model)
        if index < 0 and self._model:
            self.combo_model.addItem(self._model_display(Path(self._model)),
                                     self._model)
            index = self.combo_model.count() - 1
        self.combo_model.addItem("浏览模型文件…", "__browse__")
        self.combo_model.setCurrentIndex(max(0, index))

    def _update_eval_label(self) -> None:
        about = self._about or {}
        value = about.get("value_model") or {}
        badge = "手工评估"
        if about.get("value_model_loaded"):
            status = "候选试玩（未晋级正式版本）"
            try:
                state = self._registry.read()
                sha = file_identity(value["path"])["sha256"]
                for entry in state["versions"]:
                    if entry["sha256"] == sha:
                        status = (("当前正式版本 " if entry["version"] == state["current"]
                                   else "冻结历史版本 ") + entry["version"])
            except (ValueError, OSError, KeyError):
                pass
            badge = ("候选模型" if status.startswith("候选")
                     else status.replace("当前正式版本 ", "正式 ")
                     .replace("冻结历史版本 ", "历史 "))
            backend = str(value.get("backend", "unknown"))
            backend_text = {
                "incremental-avx2": "增量累加 + AVX2",
                "incremental-scalar": "增量累加 + 标量前向",
                "full-rebuild-scalar": "全量标量参照",
            }.get(backend, backend)
            text = (
                f"评估方式：价值网络 GMVALUE1（{backend_text}）\n"
                f"模型：{value.get('model_id', '?')}  "
                f"{value.get('arch', '')}\n"
                f"规则/特征/价值：{value.get('rule', '?')} / "
                f"{value.get('feature_spec', '?')} / "
                f"{value.get('value_contract', '?')}\n"
                f"文件：{value.get('path', '')}\n"
                "状态：" + status)
        elif about.get("nnue_loaded"):
            badge = "旧版 NNUE"
            text = ("评估方式：旧版 NNUE（" + str(about.get("nnue_backend"))
                    + "）\n文件：" + str(about.get("nnue_path")))
        else:
            text = "评估方式：手工威胁评估（没有加载已训练模型）"
        self.eval_label.setText(text)
        if hasattr(self, "header_status"):
            self.header_status.setText(badge)

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 GMVALUE1 模型", str(ROOT / "weights"),
            "GMVALUE1 模型 (*.gmvalue);;所有文件 (*)")
        if not path:
            return
        index = self.combo_model.findData(path)
        if index < 0:
            self.combo_model.addItem(self._model_display(Path(path)), path)
            index = self.combo_model.count() - 1
        self.combo_model.setCurrentIndex(index)
        self._apply_model_path(path)

    def _apply_model(self) -> None:
        data = self.combo_model.currentData()
        if data == "__browse__":
            self._browse_model()
            return
        self._apply_model_path(data)

    def _apply_model_path(self, path: str | None) -> bool:
        """Load a model into a fresh engine process; never claim a failed one.

        The old process keeps running until the new one has produced a valid
        ``about`` reply, so a rejected candidate cannot break the current game
        or silently switch the visible evaluation identity.
        """
        if self._phase in ("thinking", "collecting", "training", "evaluating", "continuous"):
            self._show_error("当前任务运行中，请先停止后再切换模型")
            return False
        candidate = str(path) if path else None
        try:
            client, about = start_executor(self._exe, candidate)
        except ExecutorError as exc:
            self._show_error(str(exc))
            return False
        old = self._client
        self._client = client
        self._about = about
        self._model = candidate
        self.history_panel.edit_candidate.setText(candidate or "")
        try:
            old.close()
        except Exception:  # noqa: BLE001 - switching models must not fail
            pass
        self._startup_error = ""
        self._phase = "idle"
        self._reload_model_choices()
        self._update_eval_label()
        self.status_label.setText(
            "模型已应用：" + (NO_MODEL_TEXT if candidate is None else candidate))
        self._new_game()
        return True

    # ---- game flow -----------------------------------------------------

    def _new_game(self) -> None:
        if self._phase in ("evaluating", "continuous"):
            self._show_error("测评运行中或持续训练运行中，请先停止任务")
            return
        if self._phase == "collecting":
            self._show_error("采集运行中，请先停止采集")
            return
        if self._phase == "training":
            self._show_error("训练运行中，请先停止训练")
            return
        # A new player game always takes the board back, even during the short
        # stopped-frame hold after a background task.
        self._live_hold_token += 1
        self._background_active = False
        self._preview = None
        self._live_forced = False
        self.btn_live.hide()
        self.play_live_hint.hide()
        self.btn_new.setEnabled(True)
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
        if self._startup_error:
            self._show_error(self._startup_error)
        self._maybe_engine_move()

    def _stop_game(self) -> None:
        if self._phase in ("evaluating", "continuous"):
            self._show_error("请用对应任务的停止按钮结束当前任务")
            return
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
        try:
            self._client, self._about = start_executor(self._exe, self._model)
        except ExecutorError as exc:
            # The selected model disappeared or no longer loads. Keep the app
            # usable but say so; never leave the old identity on screen.
            self._startup_error = str(exc)
            self._model = None
            self._client, self._about = start_executor(self._exe, None)
        self._reload_model_choices()
        self._update_eval_label()
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
        self.activity_label.setText("引擎思考中…")
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
                self._train_pause = None
                self.btn_train_start.setEnabled(True)
                self.btn_train_pause.setEnabled(False)
                self.btn_train_stop.setEnabled(False)
                if kind == "train_done":
                    paused = (payload.get("stopped") is True
                              and payload.get("stop_reason") == "paused")
                    if paused:
                        checkpoint = (payload.get("checkpoint") or {}).get("last")
                        self._train_resume = checkpoint
                        self.btn_train_resume.setEnabled(
                            bool(checkpoint and Path(checkpoint).is_file()))
                        self.status_label.setText("训练已暂停；可在安全检查点继续")
                        self.activity_label.setText("从0训练已暂停")
                    else:
                        self._train_resume = None
                        self.btn_train_resume.setEnabled(False)
                        self.status_label.setText("训练完成")
                        self.activity_label.setText("从0训练完成")
                    self._show_training_result(payload)
                else:
                    self._train_resume = None
                    self.btn_train_resume.setEnabled(False)
                    self._show_error("训练失败：" + str(payload))
                continue
            if kind in ("collect_progress", "collect_preview",
                        "collect_done", "collect_error"):
                if generation != self._collect_generation:
                    continue
                if kind == "collect_preview":
                    self._preview_paused = False
                    self._preview_training = None
                    self._on_preview(payload)
                    continue
                if kind == "collect_progress":
                    self._show_collection_update(payload)
                    continue
                self._phase = "idle"
                self._set_background(False, hold=True)
                self.activity_label.setText("准备就绪")
                self._preview_paused = False
                self._preview_training = None
                self._collect_thread = None
                self._collect_stop = None
                self._collect_pause = None
                self.btn_collect_start.setEnabled(True)
                self.btn_collect_pause.setEnabled(False)
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
        if self._state is None or self._phase not in ("idle", "game_over"):
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
        path = self._pool_path
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
        if self._phase in ("collecting", "thinking", "training", "evaluating", "continuous"):
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
            vcf_nodes=5000, vcf_plies=4,
            seed=self._collection_seed if self._collection_seed is not None else int(time.time()) % 100000,
            model_id="handcrafted")
        stop_event = threading.Event()
        pause_event = threading.Event()
        self._collect_stop = stop_event
        self._collect_pause = pause_event
        self._collect_generation += 1
        generation = self._collect_generation
        self._phase = "collecting"
        self.status_label.setText("采集运行中…")
        self.activity_label.setText("采集运行中")
        self._set_background(True)
        self.btn_collect_start.setEnabled(False)
        self.btn_collect_pause.setEnabled(True)
        self.btn_collect_stop.setEnabled(True)
        exe = self._exe
        pool_path = self._pool_path

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
                        stop_event=stop_event, pause_event=pause_event,
                        on_position=lambda payload: self._queue.put(
                            ("collect_preview", generation, payload)),
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

    def _pause_collection(self) -> None:
        if self._collect_pause is None:
            return
        self._collect_pause.set()
        self.btn_collect_pause.setEnabled(False)
        self.status_label.setText("采集暂停中：当前棋局完成后停在安全边界")

    def _stop_collection(self) -> None:
        if self._collect_stop is None:
            return
        self._collect_stop.set()
        self._collect_pause.set()  # wakes the between-games pause wait
        self.btn_collect_stop.setEnabled(False)
        self.btn_collect_pause.setEnabled(False)
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
        if not self._live_showing():
            self.activity_label.setText(
                "采集 · 已入池 " + str(update.get("inserted", 0))
                + (" · 已完成" if done else ""))
        filters = update.get("filters") or {}
        if filters:
            top = ", ".join(f"{name}={count}"
                            for name, count in sorted(filters.items()))
            self.collect_label.setText(self.collect_label.text() + "；过滤 " + top)
        if done:
            self.collect_pool_label.setText(self._pool_status_text())

    def _start_training(self, resume: str | None = None) -> None:
        if self._phase != "idle":
            return
        pool_path = self.edit_pool.text().strip()
        output_dir = self.edit_train_output.text().strip()
        if not pool_path or not output_dir:
            self._show_error("需要样本池路径和输出目录")
            return
        if resume:
            steps = self._train_target
            seed = self._train_seed
        else:
            steps = self.spin_train_steps.value()
            seed = int(time.time()) % 100000
            self._train_seed = seed
            self._train_target = steps
            self._train_resume = None
        batch = self.spin_train_batch.value()
        lr = self.spin_train_lr.value()
        loss = "mse" if self.combo_train_loss.currentIndex() == 0 else "smooth_l1"
        stop_event = threading.Event()
        pause_event = threading.Event()
        self._train_stop = stop_event
        self._train_pause = pause_event
        self._train_generation += 1
        generation = self._train_generation
        self._phase = "training"
        self.activity_label.setText("从0训练运行中")
        self.status_label.setText(
            "训练运行中…（可点「暂停训练」保存检查点）" if resume is None
            else "从检查点继续训练…")
        self.btn_train_start.setEnabled(False)
        self.btn_train_pause.setEnabled(True)
        self.btn_train_resume.setEnabled(False)
        self.btn_train_stop.setEnabled(True)
        self.train_label.setText("训练：正在加载数据与初始化…")
        self.train_result.setText("")

        def worker() -> None:
            try:
                import value_train
                config = value_train.TrainConfig(
                    pool=pool_path, output_dir=output_dir,
                    seed=seed, steps=steps,
                    batch_size=batch, lr=lr, loss=loss, device="auto",
                    resume=resume,
                    val_every=max(5, steps // 5),
                    checkpoint_every=max(10, steps // 2),
                    log_every=max(1, steps // 10))
                summary = value_train.train_value(
                    config, stop_event=stop_event, pause_event=pause_event,
                    progress=lambda update: self._queue.put(
                        ("train_progress", generation, update)))
                self._queue.put(("train_done", generation, summary))
            except Exception as exc:  # noqa: BLE001 - surfaced in the UI
                self._queue.put(("train_error", generation, str(exc)))

        self._train_thread = threading.Thread(target=worker, daemon=True)
        self._train_thread.start()

    def _pause_training(self) -> None:
        if self._train_pause is None:
            return
        self._train_pause.set()
        self.btn_train_pause.setEnabled(False)
        self.status_label.setText("训练暂停中：将在优化步边界保存检查点")

    def _resume_training(self) -> None:
        if self._phase != "idle" or not self._train_resume:
            return
        if not Path(self._train_resume).is_file():
            self._show_error("训练检查点已不存在，无法继续")
            return
        self._start_training(resume=self._train_resume)

    def _stop_training(self) -> None:
        if self._train_stop is None:
            return
        self._train_stop.set()
        if self._train_pause is not None:
            self._train_pause.set()  # wakes a paused wait so the worker exits
        self.btn_train_stop.setEnabled(False)
        self.btn_train_pause.setEnabled(False)
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
        if self._state is None or self._phase not in ("idle", "game_over"):
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
        store_path = self._teacher_store_path
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
        if self._live_showing():
            self.activity_label.setText("实时棋盘只读 · 点「返回对局」继续")
            return
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
                and self._phase != "thinking" and not self._live_showing()):
            try:
                lm = self._client.legal_moves()
                legal = {tuple(m) for m in lm.get("moves", [])}
                forbidden = {(f["x"], f["y"]): f["reason"]
                             for f in lm.get("forbidden", [])}
            except ExecutorError as exc:
                self._show_error(str(exc))
        last = tuple(state["last_move"]) if state.get("last_move") else None
        if self._live_showing():
            self._render_preview()
        else:
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
        if self._live_showing():
            return
        if self._background_active:
            self.activity_label.setText("后台任务运行中 · 点 LIVE 查看")
        elif self._phase == "thinking":
            self.activity_label.setText("引擎思考中…")
        elif state["result"] != "ongoing":
            self.activity_label.setText(
                "对局结束 · " + RESULT_TEXT.get(state["result"], state["result"]))
        elif state["side_to_move"] == self._player_color:
            self.activity_label.setText("轮到你落子")
        else:
            self.activity_label.setText("引擎行棋中…")

    def _show_error(self, message: str) -> None:
        self.error_label.setText("错误：" + message)
        self.error_label.show()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        sheet = getattr(self, "settings_sheet", None)
        if sheet is not None and sheet.visible():
            sheet.reposition()

    def closeEvent(self, event) -> None:
        self.training_task_panel.shutdown()
        self.history_panel.shutdown()
        self._generation += 1
        if self._collect_stop is not None:
            self._collect_stop.set()
        if self._collect_pause is not None:
            self._collect_pause.set()
        if self._collect_thread is not None:
            self._collect_thread.join(timeout=10)
        if self._train_stop is not None:
            self._train_stop.set()
        if self._train_pause is not None:
            self._train_pause.set()
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
        if self._restore_geometry:
            self._settings.setValue("window/geometry", self.saveGeometry())
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
    parser.add_argument("--model", help="启动时加载的 GMVALUE1 价值网络（试玩候选）")
    parser.add_argument("--pool", default=str(ROOT / "data/pool/pool.sqlite3"),
                        help="采集与训练的样本池路径")
    parser.add_argument("--collection-seed", type=int, help="可复现的采集随机种子")
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
    window = MainWindow(exe, args.movetime, model=args.model,
                        pool=args.pool, collection_seed=args.collection_seed,
                        restore_geometry=True)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
