"""Teacher, transition and teacher-free modes using the shared scheduler."""
from __future__ import annotations

from pathlib import Path
import json
import os
import queue
import random
import threading

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QTextEdit, QVBoxLayout, QWidget)

from ui_components import CollapsibleSection, mark_button


class TrainingTaskPanel(QGroupBox):
    busyChanged = Signal(bool)
    published = Signal(str)
    previewReceived = Signal(dict)
    stateChanged = Signal(dict)
    recoverableChanged = Signal(int)

    def __init__(self, exe, pool, registry, evaluations, output_root, can_start):
        super().__init__("运行")
        self.exe, self.pool, self.registry = exe, pool, registry
        self.evaluations, self.output_root = evaluations, output_root
        self.default_outputs = {"teacher": str(output_root),
            "transition": str(Path(output_root).parent / "transition-tasks"),
            "selfplay": str(Path(output_root).parent / "selfplay-tasks")}
        self.can_start = can_start
        self.messages = queue.Queue()
        self.task = self.thread = self.last_result = None
        self._recoverable = []
        layout = QVBoxLayout(self)
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("教师训练（新增 Rapfi 标签）", "teacher")
        self.combo_mode.addItem("自主过渡（保留历史教师样本）", "transition")
        self.combo_mode.addItem("完全无教师（仅真实胜负）", "selfplay")
        layout.addWidget(self.combo_mode)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setObjectName("Caption")
        layout.addWidget(self.hint)
        settings_layout = QVBoxLayout()
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(6)
        row = QHBoxLayout()
        row.addWidget(QLabel("任务输出"))
        self.edit_output = QLineEdit(str(output_root))
        row.addWidget(self.edit_output)
        settings_layout.addLayout(row)
        self.spin_capacity = self._spin(64, 1000000, 1000)
        self.spin_min_train = self._spin(1, 100000, 32)
        self.spin_new = self._spin(1, 100000, 32)
        self.spin_steps = self._spin(1, 100000, 200)
        self.spin_batch = self._spin(1, 2048, 32)
        self.spin_workers = self._spin(1, 256, 1)
        self.spin_eval_workers = self._spin(1, 256, 1)
        self.spin_cpu_threads = self._spin(1, 256, 2)
        self.spin_sample_limit = self._spin(1, 1000000, 512)
        self.spin_pending = self._spin(1, 1000000, 128)
        self.spin_validation_games = self._spin(1, 10000, 2)
        self.spin_state_seconds = self._spin(1, 60, 2)
        self.spin_checkpoint_seconds = self._spin(1, 3600, 60)
        self.spin_rounds = self._spin(1, 10000, 10)
        self.spin_games = self._spin(1, 100000, 1000)
        self.spin_minutes = self._spin(1, 1440, 60)
        self.spin_pairs = self._spin(1, 10000, 64)
        self.spin_ms = self._spin(10, 60000, 200)
        for fields in (
            (("池容量", self.spin_capacity), ("启动训练样本", self.spin_min_train), ("新增样本/轮", self.spin_new)),
            (("优化步/轮", self.spin_steps), ("批", self.spin_batch), ("最多候选", self.spin_rounds)),
            (("最多采集局", self.spin_games), ("最长分钟", self.spin_minutes)),
            (("采集并发", self.spin_workers), ("测评并发", self.spin_eval_workers), ("训练 CPU 线程", self.spin_cpu_threads)),
            (("训练快照上限", self.spin_sample_limit), ("待训练积压", self.spin_pending), ("验证对局", self.spin_validation_games)),
            (("状态写入间隔秒", self.spin_state_seconds), ("检查点最小间隔秒", self.spin_checkpoint_seconds)),
            (("测评开局对", self.spin_pairs), ("每步 ms", self.spin_ms))):
            row = QHBoxLayout()
            for label, widget in fields:
                row.addWidget(QLabel(label))
                row.addWidget(widget)
            settings_layout.addLayout(row)
        row = QHBoxLayout()
        self.btn_throughput = QPushButton("高吞吐预设（本机）")
        self.btn_throughput.clicked.connect(self._high_throughput)
        mark_button(self.btn_throughput, "quiet")
        self.btn_quick = QPushButton("少量数据快速启动")
        self.btn_quick.clicked.connect(self._quick_start)
        mark_button(self.btn_quick, "quiet")
        row.addWidget(self.btn_throughput)
        row.addWidget(self.btn_quick)
        settings_layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("训练设备"))
        self.combo_device = QComboBox()
        self.combo_device.addItems(["auto", "cpu", "cuda"])
        row.addWidget(self.combo_device)
        row.addWidget(QLabel("学习率"))
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setDecimals(5)
        self.spin_lr.setRange(0.00001, 1)
        self.spin_lr.setValue(0.001)
        row.addWidget(self.spin_lr)
        settings_layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("自对弈每步 ms"))
        self.spin_selfplay_ms = self._spin(1, 60000, 30)
        row.addWidget(self.spin_selfplay_ms)
        row.addWidget(QLabel("探索概率"))
        self.spin_epsilon = QDoubleSpinBox()
        self.spin_epsilon.setRange(0, 1)
        self.spin_epsilon.setSingleStep(.05)
        self.spin_epsilon.setValue(.1)
        row.addWidget(self.spin_epsilon)
        row.addWidget(QLabel("历史教师权重"))
        self.spin_teacher_weight = QDoubleSpinBox()
        self.spin_teacher_weight.setRange(.01, 1)
        self.spin_teacher_weight.setValue(.25)
        row.addWidget(self.spin_teacher_weight)
        settings_layout.addLayout(row)
        self.settings = CollapsibleSection("任务与资源设置", settings_layout)
        self.settings.embed()
        row = QHBoxLayout()
        self.btn_start = QPushButton("启动教师训练任务")
        self.btn_start.clicked.connect(self.start)
        self.btn_pause = QPushButton("暂停任务")
        self.btn_pause.clicked.connect(self.pause)
        self.btn_pause.setEnabled(False)
        self.btn_resume = QPushButton("继续暂停的任务")
        self.btn_resume.clicked.connect(self.resume_runtime)
        self.btn_resume.setEnabled(False)
        self.btn_stop = QPushButton("停止任务")
        self.btn_stop.clicked.connect(self.stop)
        self.btn_stop.setEnabled(False)
        mark_button(self.btn_start, "primary")
        mark_button(self.btn_stop, "danger")
        for button in (self.btn_start, self.btn_pause, self.btn_resume,
                       self.btn_stop):
            row.addWidget(button)
        layout.addLayout(row)
        self.recovery_widget = QWidget()
        recovery_layout = QVBoxLayout(self.recovery_widget)
        recovery_layout.setContentsMargins(0, 0, 0, 0)
        recovery_layout.setSpacing(6)
        row = QHBoxLayout()
        row.addWidget(QLabel("恢复已中断任务"))
        self.combo_recover = QComboBox()
        row.addWidget(self.combo_recover, 1)
        self.btn_refresh_tasks = QPushButton("刷新")
        self.btn_refresh_tasks.clicked.connect(self._refresh_recoverable)
        mark_button(self.btn_refresh_tasks, "quiet")
        row.addWidget(self.btn_refresh_tasks)
        self.btn_recover = QPushButton("恢复所选任务")
        self.btn_recover.clicked.connect(self.resume_selected)
        row.addWidget(self.btn_recover)
        recovery_layout.addLayout(row)
        self.btn_bootstrap = QPushButton("先建立 v0：切换到教师训练")
        self.btn_bootstrap.clicked.connect(self._choose_teacher)
        mark_button(self.btn_bootstrap, "quiet")
        layout.addWidget(self.btn_bootstrap)
        self.status = QLabel("任务：未开始")
        self.status.setWordWrap(True)
        self.status.setObjectName("Caption")
        layout.addWidget(self.status)
        self.metrics = QTextEdit()
        self.metrics.setReadOnly(True)
        self.metrics.setMaximumHeight(260)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(50)
        self.combo_mode.currentIndexChanged.connect(self._mode_changed)
        self._high_throughput()
        self._mode_changed()
        self._refresh_recoverable()

    def _high_throughput(self):
        cpus = os.cpu_count() or 1
        evaluation = max(1, min(8, cpus // 4))
        self.spin_workers.setValue(max(1, min(64, cpus - evaluation)))
        self.spin_eval_workers.setValue(evaluation)
        self.spin_cpu_threads.setValue(max(1, min(4, cpus // 8)))
        self.spin_capacity.setValue(100000)
        self.spin_min_train.setValue(2048)
        self.spin_new.setValue(2048)
        self.spin_batch.setValue(2048)
        self.spin_sample_limit.setValue(32768)
        self.spin_pending.setValue(8192)
        self.spin_validation_games.setValue(8)

    def _quick_start(self):
        for widget, value in ((self.spin_workers, 1), (self.spin_eval_workers, 1),
                (self.spin_cpu_threads, 2), (self.spin_capacity, 1000),
                (self.spin_min_train, 32), (self.spin_new, 32), (self.spin_batch, 32),
                (self.spin_sample_limit, 512), (self.spin_pending, 128), (self.spin_validation_games, 2)):
            widget.setValue(value)

    def _mode_changed(self):
        mode = self.combo_mode.currentData()
        if self.edit_output.text() in self.default_outputs.values():
            self.edit_output.setText(self.default_outputs[mode])
        self._refresh_readiness()
        self.btn_start.setText("启动教师训练任务" if mode == "teacher" else "启动自主训练任务")
        self.spin_teacher_weight.setEnabled(mode == "transition")
        self.spin_epsilon.setEnabled(mode != "teacher")
        self.spin_selfplay_ms.setEnabled(mode != "teacher")

    def _refresh_readiness(self):
        from model_registry import ModelRegistry
        mode = self.combo_mode.currentData()
        hint = {
            "teacher": "采集 → Rapfi 打标 → 候选训练 → 历史测评。尚无模型时可建立 v0；晋级仅影响下一盘。",
            "transition": "停止新增教师标签；正式版本自对弈得到真实胜负，允许按显式权重混合历史教师样本。需要已有正式版本，这还不是完全无教师更新。",
            "selfplay": "完全无教师：需要已有正式版本，仅训练/验证完整自对弈的真实胜负；混合池中的教师标签被排除。Rapfi 缺失也可运行，不会自动切回教师。"}[mode]
        ready, missing = True, False
        if mode != "teacher":
            try:
                missing = ModelRegistry(self.registry).resolve() is None
                if missing:
                    ready = False
                    hint = ("请先建立正式版本 v0：切换到「教师训练」，点击「启动教师训练任务」，"
                            "等待正式版本显示 v0 后停止任务，再选择自主模式。"
                            "如已有合格候选，也可在「历史版本测评与晋级」中验证并登记 v0。"
                            f"\n当前登记目录：{self.registry}")
            except (OSError, ValueError) as exc:
                ready = False
                hint = f"正式版本校验失败，请先在「历史版本测评与晋级」中检查记录与模型：{exc}"
        self.hint.setText(hint)
        self.btn_bootstrap.setVisible(missing)
        self.btn_bootstrap.setEnabled(self.thread is None)
        return ready

    def _choose_teacher(self):
        self.combo_mode.setCurrentIndex(self.combo_mode.findData("teacher"))
        self.status.setText("已选择教师训练；点击「启动教师训练任务」建立首个正式版本 v0。")

    @staticmethod
    def _spin(low, high, value):
        spin = QSpinBox()
        spin.setRange(low, high)
        spin.setValue(value)
        return spin

    def make_config(self):
        from training_task import TaskConfig
        return TaskConfig(engine=str(self.exe), pool=str(self.pool), registry=str(self.registry),
            mode=self.combo_mode.currentData(), teacher_weight=self.spin_teacher_weight.value(),
            selfplay={"movetime_ms": self.spin_selfplay_ms.value(), "epsilon": self.spin_epsilon.value(),
                      "game_timeout_s": 180.0 if self.spin_workers.value() > 1 else 60.0},
            evaluations=str(self.evaluations), output_root=self.edit_output.text().strip(),
            capacity=self.spin_capacity.value(), min_train_samples=self.spin_min_train.value(),
            min_new_samples=self.spin_new.value(), max_pending_samples=max(self.spin_pending.value(), self.spin_new.value()),
            training_sample_limit=max(self.spin_sample_limit.value(), self.spin_min_train.value()),
            validation_games=self.spin_validation_games.value(), collector_workers=self.spin_workers.value(),
            cpu_threads=self.spin_cpu_threads.value(),
            state_write_interval_s=self.spin_state_seconds.value(),
            checkpoint_min_interval_s=self.spin_checkpoint_seconds.value(),
            collection_delay_ms=0 if self.spin_workers.value() > 1 else 100,
            full_pool_delay_ms=50 if self.spin_workers.value() > 1 else 500,
            min_round_interval_s=0 if self.spin_workers.value() > 1 else 10,
            train_steps=self.spin_steps.value(), batch_size=self.spin_batch.value(),
            lr=self.spin_lr.value(), device=self.combo_device.currentText(),
            max_rounds=self.spin_rounds.value(), max_games=self.spin_games.value(),
            max_minutes=self.spin_minutes.value(),
            seed=random.SystemRandom().randrange(2**31),
            evaluation={"pairs": self.spin_pairs.value(), "movetime_ms": self.spin_ms.value(),
                        "workers": self.spin_eval_workers.value()})

    def _start_worker(self, task, message: str):
        self.task = task
        task.preview = lambda payload: self.messages.put(("preview", payload))
        self.last_result = None
        self.busyChanged.emit(True)
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_resume.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_recover.setEnabled(False)
        self.status.setText(message)
        self._enable_settings(False)

        def worker():
            try:
                self.messages.put(("done", task.run()))
            except Exception as exc:
                self.messages.put(("error", str(exc)))
        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()

    def start(self):
        if self.thread is not None or not self.can_start():
            self.status.setText("任务：请先停止其它任务或对局思考")
            return
        from training_task import TrainingTask
        config = self.make_config()
        if not self._refresh_readiness():
            self.status.setText(self.hint.text())
            return
        self._start_worker(
            TrainingTask(config, progress=lambda value: self.messages.put(("state", value))),
            "任务：正在核验依赖与冻结配置…")

    def pause(self):
        if self.task is None or self.thread is None:
            return
        if self.task.pause():
            self.status.setText("任务：暂停中，等待当前棋局/优化步/测评配对到达安全边界…")
            self.btn_pause.setEnabled(False)
            self.btn_resume.setEnabled(True)

    def resume_runtime(self):
        if self.task is None or self.thread is None:
            return
        if self.task.resume():
            self.status.setText("任务：恢复中…")
            self.btn_resume.setEnabled(False)
            self.btn_pause.setEnabled(True)

    def _refresh_recoverable(self):
        roots = {Path(self.edit_output.text().strip() or str(self.output_root))}
        roots.update(Path(path) for path in self.default_outputs.values())
        found = {}
        for root in roots:
            if not root.is_dir():
                continue
            for manifest in root.glob("*/manifest.json"):
                directory = manifest.parent
                status = "unknown"
                try:
                    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
                    status = str(state.get("status", "unknown"))
                except (OSError, ValueError):
                    pass
                if status in ("completed", "failed"):
                    continue
                found[str(directory.resolve())] = status
        self._recoverable = sorted(found.items())
        selected = self.combo_recover.currentData()
        self.combo_recover.clear()
        for path, status in self._recoverable:
            self.combo_recover.addItem(f"{Path(path).name[:12]}… [{status}]", path)
        if selected:
            index = self.combo_recover.findData(selected)
            if index >= 0:
                self.combo_recover.setCurrentIndex(index)
        self.btn_recover.setEnabled(bool(self._recoverable)
                                    and self.thread is None
                                    and self.can_start())
        self.recoverableChanged.emit(len(self._recoverable))

    def resume_selected(self):
        if self.thread is not None or not self.can_start():
            self.status.setText("任务：请先停止其它任务或对局思考")
            return
        path = self.combo_recover.currentData()
        if not path:
            self.status.setText("任务：没有可恢复的任务目录")
            return
        from training_task import TrainingTask
        try:
            task = TrainingTask.resume_from_directory(
                path, progress=lambda value: self.messages.put(("state", value)))
        except (OSError, ValueError, KeyError) as exc:
            self.status.setText("恢复失败：" + str(exc))
            return
        self._start_worker(task, "任务：正在核验冻结配置、模型与检查点…")

    def _enable_settings(self, enabled):
        self.btn_throughput.setEnabled(enabled)
        self.btn_quick.setEnabled(enabled)
        widgets = (self.findChildren(QSpinBox) + self.findChildren(QDoubleSpinBox)
                   + self.findChildren(QComboBox) + self.findChildren(QLineEdit)
                   + self.settings.findChildren(QSpinBox)
                   + self.settings.findChildren(QDoubleSpinBox)
                   + self.settings.findChildren(QComboBox)
                   + self.settings.findChildren(QLineEdit)
                   + self.recovery_widget.findChildren(QComboBox)
                   + self.recovery_widget.findChildren(QPushButton))
        for widget in widgets:
            widget.setEnabled(enabled)
        self.btn_quick.setEnabled(enabled)
        self.btn_throughput.setEnabled(enabled)

    def stop(self):
        if self.task:
            self.task.stop()
            self.status.setText("任务：停止中，正在保留检查点并回收进程…")
            self.btn_stop.setEnabled(False)
            self.btn_pause.setEnabled(False)
            self.btn_resume.setEnabled(False)

    STATUS_TEXT = {"created": "未开始", "running": "运行中", "pausing": "暂停中",
                   "paused": "已暂停", "resuming": "恢复中", "stopping": "停止中",
                   "stopped": "已停止", "completed": "已完成", "failed": "失败"}

    def _sync_transport_buttons(self, status):
        busy = status in ("running", "pausing", "paused", "resuming", "stopping")
        self.btn_pause.setEnabled(busy and status in ("running", "resuming"))
        self.btn_resume.setEnabled(busy and status in ("paused", "pausing"))
        self.btn_stop.setEnabled(busy)
        if not busy:
            self.btn_recover.setEnabled(bool(self._recoverable)
                                        and self.thread is None
                                        and self.can_start())

    def show_state(self, state):
        data, train, evaluation = (state[name] for name in ("data", "trainer", "evaluation"))
        status = self.STATUS_TEXT.get(state['status'], state['status'])
        self.status.setText(f"任务：{status}；采集 {data.get('phase', '-')}；训练 {train.get('phase', '-')}；测评 {evaluation.get('phase', '-')}")
        self._sync_transport_buttons(state['status'])
        metrics = data.get("totals", data.get("metrics", {}))
        progress = train.get("progress", {})
        current = (evaluation.get("result", {}).get("publication") or state.get("current") or {})
        lines = [f"模式 {state.get('mode', 'teacher')}，训练来源 {state.get('training_sources')}，权重 {state.get('source_weights')}",
            f"采集局数 {state['games']}，候选轮数 {len(state['rounds'])}",
            f"池累计生成 {metrics.get('generated', 0)}，有效 {metrics.get('valid', 0)}，入池 {metrics.get('inserted', 0)}",
            f"池容量 {metrics.get('total_pool_samples', metrics.get('pool_samples', 0))}/{metrics.get('capacity', '?')}，可用 {metrics.get('pool_samples', 0)}，有效/秒 {metrics.get('effective_per_second', 0):.2f}",
            f"采集模型 {data.get('version') or data.get('model_id', 'handcrafted')}",
            f"训练 step {progress.get('step', '-')}，loss {progress.get('train_loss', '-')}，val MAE {progress.get('val_mae', '-')}，设备 {train.get('summary', {}).get('device', state['resources'].get('device'))}",
            f"正式版本 {current.get('version', '尚未建立')}；资源 {state['resources']}",
            f"测评结果 {evaluation.get('result', {}).get('statistics', {})}",
            f"结束原因 {state.get('reason')}；任务记录 {state['directory']}"]
        self.metrics.setPlainText("\n".join(lines))

    def poll(self):
        while True:
            try:
                kind, state = self.messages.get_nowait()
            except queue.Empty:
                return
            if kind == "preview":
                self.previewReceived.emit(state)
                continue
            if kind != "error":
                self.show_state(state)
            if kind == "state":
                self.stateChanged.emit(state)
                continue
            self.thread = None
            self.btn_start.setEnabled(True)
            self.btn_pause.setEnabled(False)
            self.btn_resume.setEnabled(False)
            self.btn_stop.setEnabled(False)
            self._enable_settings(True)
            self._mode_changed()
            self._refresh_recoverable()
            self.busyChanged.emit(False)
            if kind == "error":
                self.status.setText("任务失败：" + state)
            else:
                self.last_result = state
                if state.get("current"):
                    self.published.emit(state["current"]["path"])

    def shutdown(self):
        self.stop()
        if self.thread:
            self.thread.join(timeout=30)
        self.timer.stop()
