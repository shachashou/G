"""GUI adapter for the shared history evaluation backend; no game rules."""
from __future__ import annotations

import json
from pathlib import Path
import queue
import random
import threading

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QDoubleSpinBox, QFileDialog, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QPushButton, QSpinBox, QTextEdit, QVBoxLayout,
)
from model_registry import ModelRegistry
from ui_components import CollapsibleSection, mark_button


class HistoryPanel(QGroupBox):
    busyChanged = Signal(bool)
    published = Signal(str)

    def __init__(self, exe, registry, output_root, can_start):
        super().__init__("测评与晋级")
        self.exe, self.registry, self.output_root = exe, registry, output_root
        self.can_start = can_start
        self.thread = None
        self.stop_event = None
        self.last_result = None
        self._resume_run = None
        self.messages = queue.Queue()
        layout = QVBoxLayout(self)
        self.current_label = QLabel()
        self.current_label.setWordWrap(True)
        layout.addWidget(self.current_label)
        row = QHBoxLayout()
        self.edit_candidate = QLineEdit()
        self.edit_candidate.setPlaceholderText("候选 model.gmvalue（同目录需保真度和训练局面清单）")
        row.addWidget(self.edit_candidate)
        browse = QPushButton("候选…")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        layout.addLayout(row)
        advanced_layout = QVBoxLayout()
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(6)
        advanced_layout.addWidget(QLabel("额外历史对手（可多选；当前正式版本自动加入）"))
        self.opponents = QListWidget()
        self.opponents.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.opponents.setMaximumHeight(80)
        advanced_layout.addWidget(self.opponents)
        row = QHBoxLayout()
        row.addWidget(QLabel("开局对数"))
        self.spin_pairs = QSpinBox()
        self.spin_pairs.setRange(1, 10000)
        self.spin_pairs.setValue(64)
        row.addWidget(self.spin_pairs)
        row.addWidget(QLabel("每步 ms"))
        self.spin_ms = QSpinBox()
        self.spin_ms.setRange(10, 60000)
        self.spin_ms.setValue(200)
        row.addWidget(self.spin_ms)
        row.addWidget(QLabel("并发对局"))
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(1, 256)
        self.spin_workers.setValue(1)
        row.addWidget(self.spin_workers)
        row.addWidget(QLabel("得分率基线"))
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.5, 0.99)
        self.spin_threshold.setValue(0.5)
        self.spin_threshold.setSingleStep(0.01)
        row.addWidget(self.spin_threshold)
        advanced_layout.addLayout(row)
        hint = QLabel("赛前冻结；同开局换色、各 1 线程。至少 32 对；配对单侧 95% 下界须高于基线，多个对手校正。异常即停止且不晋级。v0 只验证一对完整合法对局，不宣称棋力提升。")
        hint.setWordWrap(True)
        hint.setObjectName("Caption")
        advanced_layout.addWidget(hint)
        self.advanced = CollapsibleSection("测评参数与对手", advanced_layout)
        layout.addWidget(self.advanced)
        self.btn_initial = QPushButton("登记 v0")
        self.btn_initial.clicked.connect(lambda: self.start(True))
        self.btn_challenge = QPushButton("开始测评")
        self.btn_challenge.clicked.connect(lambda: self.start(False))
        self.btn_stop = QPushButton("停止测评")
        self.btn_stop.clicked.connect(self.stop)
        self.btn_stop.setEnabled(False)
        self.btn_pause = QPushButton("暂停测评")
        self.btn_pause.clicked.connect(self.pause)
        self.btn_pause.setEnabled(False)
        self.btn_resume_eval = QPushButton("继续上次中断的测评")
        self.btn_resume_eval.clicked.connect(self.resume_interrupted)
        self.btn_resume_eval.setEnabled(False)
        mark_button(self.btn_initial, "primary")
        mark_button(self.btn_challenge, "primary")
        mark_button(self.btn_stop, "danger")
        primary_row = QHBoxLayout()
        primary_row.setSpacing(8)
        primary_row.addWidget(self.btn_initial)
        primary_row.addWidget(self.btn_challenge)
        layout.addLayout(primary_row)
        secondary_row = QHBoxLayout()
        secondary_row.setSpacing(8)
        for button in (self.btn_pause, self.btn_resume_eval, self.btn_stop):
            secondary_row.addWidget(button)
        layout.addLayout(secondary_row)
        self.status = QLabel("测评：未开始")
        self.status.setWordWrap(True)
        self.status.setObjectName("Caption")
        layout.addWidget(self.status)
        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(170)
        layout.addWidget(self.report)
        row = QHBoxLayout()
        self.btn_current = QPushButton("使用当前正式版本")
        self.btn_current.clicked.connect(self.use_current)
        row.addWidget(self.btn_current)
        view = QPushButton("查看已有报告…")
        view.clicked.connect(self.view_report)
        row.addWidget(view)
        layout.addLayout(row)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(40)
        self.refresh()

    def refresh(self):
        try:
            store = ModelRegistry(self.registry)
            state = store.read()
            current = store.resolve()
            self.current_label.setText("正式版本：" + (
                f"{current['version']} / {current['model_id']}" if current else "尚未建立"))
            selected = {i.text() for i in self.opponents.selectedItems()}
            self.opponents.clear()
            for entry in state["versions"]:
                self.opponents.addItem(entry["version"])
                if entry["version"] in selected:
                    self.opponents.item(self.opponents.count() - 1).setSelected(True)
            busy = self.thread is not None
            for widget in (self.edit_candidate, self.opponents, self.spin_pairs,
                           self.spin_ms, self.spin_threshold):
                widget.setEnabled(not busy)
            self.btn_initial.setEnabled(not busy and current is None)
            self.btn_challenge.setEnabled(not busy and current is not None)
            self.btn_current.setEnabled(not busy and current is not None)
            self.btn_pause.setEnabled(busy)
            self.btn_resume_eval.setEnabled(not busy and self._resume_run is not None)
            self.btn_stop.setEnabled(busy)
        except (ValueError, OSError) as exc:
            self.current_label.setText("正式版本不可用：" + str(exc))
            self.btn_initial.setEnabled(False)
            self.btn_challenge.setEnabled(False)
            self.btn_current.setEnabled(False)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择候选模型", "weights", "GMVALUE1 (*.gmvalue)")
        if path:
            self.edit_candidate.setText(path)

    def start(self, initial=False, resume_run=None):
        if self.thread is not None or not self.can_start():
            self.status.setText("测评：请先停止当前任务或对局思考")
            return
        candidate = self.edit_candidate.text().strip()
        if not candidate and resume_run is None:
            self.status.setText("测评：请选择候选模型")
            return
        # Capture widget values on the GUI thread, before starting the worker.
        settings = {"pairs": 1 if initial else self.spin_pairs.value(),
                    "workers": self.spin_workers.value(),
                    "movetime_ms": self.spin_ms.value(),
                    "score_threshold": self.spin_threshold.value(),
                    "seed": random.SystemRandom().randrange(2**31)}
        opponents = [item.text() for item in self.opponents.selectedItems()]
        stop = self.stop_event = threading.Event()
        self.last_result = None
        self.busyChanged.emit(True)
        self.status.setText("测评：冻结身份、开局和配置中…"
                            if resume_run is None else "测评：按已冻结开局重做中断的对局…")
        self.report.clear()
        self.btn_stop.setEnabled(True)
        self.btn_pause.setEnabled(True)
        self.btn_resume_eval.setEnabled(False)

        def worker():
            try:
                from history_evaluation import EvaluationConfig, evaluate
                config = EvaluationConfig(**settings)
                kwargs = dict(registry=self.registry,
                    output_root=self.output_root, config=config,
                    opponents=[] if resume_run is not None else opponents,
                    bootstrap=initial, stop_event=stop,
                    progress=lambda item: self.messages.put(("progress", item)))
                if resume_run is not None:
                    kwargs["resume_run"] = resume_run
                result = evaluate(candidate or ".", self.exe, **kwargs)
                self.messages.put(("done", result))
            except Exception as exc:
                self.messages.put(("error", str(exc)))

        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()
        self.refresh()

    def pause(self):
        if self.stop_event is not None:
            self.stop_event.set()
            self.status.setText("测评：暂停中；当前配对结果丢弃，恢复时按同一冻结开局重做")
            self.btn_pause.setEnabled(False)

    def resume_interrupted(self):
        if self._resume_run is None:
            self.status.setText("测评：没有可继续的中断报告")
            return
        self.start(False, resume_run=self._resume_run)

    def stop(self):
        if self.stop_event is not None:
            self.stop_event.set()
            self.status.setText("测评：停止中；保留原正式版本")
            self.btn_stop.setEnabled(False)
            self.btn_pause.setEnabled(False)

    def poll(self):
        while True:
            try:
                kind, item = self.messages.get_nowait()
            except queue.Empty:
                return
            if kind == "progress":
                self.status.setText(f"测评：{item['opponent']}，{item['games']}/{item['total']} 局")
                continue
            self.thread = None
            self.stop_event = None
            self.btn_stop.setEnabled(False)
            self.btn_pause.setEnabled(False)
            self.busyChanged.emit(False)
            if kind == "error":
                self.status.setText("测评失败：" + item)
            else:
                self.last_result = item
                if item.get("status") == "interrupted":
                    self._resume_run = item.get("run")
                elif item.get("status") == "completed":
                    self._resume_run = None
                self.show_report(item)
                if item.get("publication"):
                    self.published.emit(item["publication"]["path"])
            self.refresh()

    def show_report(self, result):
        publication = result.get("publication")
        label = ("已登记 v0（仅合法性与数值验证）" if result.get("initial_registration_only")
                 else "已晋级 " + publication["version"]) if publication else "保留原正式版本"
        self.status.setText("测评：" + label + "；" + result.get("status", "未知"))
        summaries = {name: {"胜": s["wins"], "和": s["draws"], "负": s["losses"],
            "有效局数": s["valid_games"], "严格胜率": s["strict_win_rate"],
            "得分率": s["score_rate"], "配对下界": s["one_sided_lower"],
            "结论": s["conclusion"], "异常": s["anomalies"]}
            for name, s in result.get("statistics", {}).items()}
        self.report.setPlainText(json.dumps({"对手报告": summaries,
            "用时秒": result.get("elapsed_s"), "报告目录": result.get("run"),
            "发布错误": result.get("publication_error")}, ensure_ascii=False, indent=2))

    def use_current(self):
        if self.can_start():
            try:
                entry = ModelRegistry(self.registry).resolve()
                if entry:
                    self.published.emit(entry["path"])
            except (ValueError, OSError) as exc:
                self.status.setText("正式版本加载失败：" + str(exc))

    def view_report(self):
        path, _ = QFileDialog.getOpenFileName(self, "查看测评报告", str(self.output_root), "JSON (*.json)")
        if path:
            try:
                result = json.loads(Path(path).read_text(encoding="utf-8"))
                receipt = Path(path).with_name("result.json")
                if Path(path).name == "report.json" and receipt.is_file():
                    from run_support import file_identity
                    published = json.loads(receipt.read_text(encoding="utf-8"))
                    if published.get("report_sha256") == file_identity(path)["sha256"]:
                        result = published
                self.show_report(result)
            except (ValueError, OSError, KeyError) as exc:
                self.status.setText("报告不可读：" + str(exc))

    def shutdown(self):
        self.stop()
        if self.thread is not None:
            self.thread.join(timeout=20)
        self.timer.stop()
