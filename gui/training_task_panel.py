"""Teacher, transition and teacher-free modes using the shared scheduler."""
from __future__ import annotations

from pathlib import Path
import queue
import random
import threading

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QTextEdit, QVBoxLayout)


class TrainingTaskPanel(QGroupBox):
    busyChanged = Signal(bool)
    published = Signal(str)

    def __init__(self, exe, pool, registry, evaluations, output_root, can_start):
        super().__init__("持续训练：教师 / 自主")
        self.exe, self.pool, self.registry = exe, pool, registry
        self.evaluations, self.output_root = evaluations, output_root
        self.default_outputs = {"teacher": str(output_root),
            "transition": str(Path(output_root).parent / "transition-tasks"),
            "selfplay": str(Path(output_root).parent / "selfplay-tasks")}
        self.can_start = can_start
        self.messages = queue.Queue()
        self.task = self.thread = self.last_result = None
        layout = QVBoxLayout(self)
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("教师训练（新增 Rapfi 标签）", "teacher")
        self.combo_mode.addItem("自主过渡（保留历史教师样本）", "transition")
        self.combo_mode.addItem("完全无教师（仅真实胜负）", "selfplay")
        layout.addWidget(self.combo_mode)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        row = QHBoxLayout()
        row.addWidget(QLabel("任务输出"))
        self.edit_output = QLineEdit(str(output_root))
        row.addWidget(self.edit_output)
        layout.addLayout(row)
        self.spin_capacity = self._spin(64, 1000000, 1000)
        self.spin_min_train = self._spin(1, 100000, 32)
        self.spin_new = self._spin(1, 100000, 32)
        self.spin_steps = self._spin(1, 100000, 200)
        self.spin_batch = self._spin(1, 2048, 32)
        self.spin_rounds = self._spin(1, 10000, 10)
        self.spin_games = self._spin(1, 100000, 1000)
        self.spin_minutes = self._spin(1, 1440, 60)
        self.spin_pairs = self._spin(1, 10000, 64)
        self.spin_ms = self._spin(10, 60000, 200)
        for fields in (
            (("池容量", self.spin_capacity), ("启动训练样本", self.spin_min_train), ("新增样本/轮", self.spin_new)),
            (("优化步/轮", self.spin_steps), ("批", self.spin_batch), ("最多候选", self.spin_rounds)),
            (("最多采集局", self.spin_games), ("最长分钟", self.spin_minutes)),
            (("测评开局对", self.spin_pairs), ("每步 ms", self.spin_ms))):
            row = QHBoxLayout()
            for label, widget in fields:
                row.addWidget(QLabel(label))
                row.addWidget(widget)
            layout.addLayout(row)
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
        layout.addLayout(row)
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
        layout.addLayout(row)
        row = QHBoxLayout()
        self.btn_start = QPushButton("启动教师训练任务")
        self.btn_start.clicked.connect(self.start)
        self.btn_stop = QPushButton("停止任务")
        self.btn_stop.clicked.connect(self.stop)
        self.btn_stop.setEnabled(False)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop)
        layout.addLayout(row)
        self.status = QLabel("任务：未开始")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.metrics = QTextEdit()
        self.metrics.setReadOnly(True)
        self.metrics.setMaximumHeight(180)
        layout.addWidget(self.metrics)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(50)
        self.combo_mode.currentIndexChanged.connect(self._mode_changed)
        self._mode_changed()

    def _mode_changed(self):
        mode = self.combo_mode.currentData()
        if self.edit_output.text() in self.default_outputs.values():
            self.edit_output.setText(self.default_outputs[mode])
        self.hint.setText({
            "teacher": "采集 → Rapfi 打标 → 候选训练 → 历史测评。尚无模型时可建立 v0；晋级仅影响下一盘。",
            "transition": "停止新增教师标签；正式版本自对弈得到真实胜负，允许按显式权重混合历史教师样本。需要已有正式版本，这还不是完全无教师更新。",
            "selfplay": "完全无教师：需要已有正式版本，仅训练/验证完整自对弈的真实胜负；混合池中的教师标签被排除。Rapfi 缺失也可运行，不会自动切回教师。"}[mode])
        self.btn_start.setText("启动教师训练任务" if mode == "teacher" else "启动自主训练任务")
        self.spin_teacher_weight.setEnabled(mode == "transition")
        self.spin_epsilon.setEnabled(mode != "teacher")
        self.spin_selfplay_ms.setEnabled(mode != "teacher")

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
            selfplay={"movetime_ms": self.spin_selfplay_ms.value(), "epsilon": self.spin_epsilon.value()},
            evaluations=str(self.evaluations), output_root=self.edit_output.text().strip(),
            capacity=self.spin_capacity.value(), min_train_samples=self.spin_min_train.value(),
            min_new_samples=self.spin_new.value(), max_pending_samples=max(128, self.spin_new.value()),
            training_sample_limit=max(512, self.spin_min_train.value()),
            train_steps=self.spin_steps.value(), batch_size=self.spin_batch.value(),
            lr=self.spin_lr.value(), device=self.combo_device.currentText(),
            max_rounds=self.spin_rounds.value(), max_games=self.spin_games.value(),
            max_minutes=self.spin_minutes.value(),
            seed=random.SystemRandom().randrange(2**31),
            evaluation={"pairs": self.spin_pairs.value(), "movetime_ms": self.spin_ms.value()})

    def start(self):
        if self.thread is not None or not self.can_start():
            self.status.setText("任务：请先停止其它任务或对局思考")
            return
        from training_task import TrainingTask
        config = self.make_config()
        self.task = TrainingTask(config, progress=lambda value: self.messages.put(("state", value)))
        task = self.task
        self.last_result = None
        self.busyChanged.emit(True)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status.setText("任务：正在核验依赖与冻结配置…")
        self._enable_settings(False)

        def worker():
            try:
                self.messages.put(("done", task.run()))
            except Exception as exc:
                self.messages.put(("error", str(exc)))
        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()

    def _enable_settings(self, enabled):
        for widget in self.findChildren(QSpinBox) + self.findChildren(QDoubleSpinBox) + self.findChildren(QComboBox) + self.findChildren(QLineEdit):
            widget.setEnabled(enabled)

    def stop(self):
        if self.task:
            self.task.stop()
            self.status.setText("任务：停止中，正在保留检查点并回收进程…")
            self.btn_stop.setEnabled(False)

    def show_state(self, state):
        data, train, evaluation = (state[name] for name in ("data", "trainer", "evaluation"))
        self.status.setText(f"任务：{state['status']}；采集 {data.get('phase', '-')}；训练 {train.get('phase', '-')}；测评 {evaluation.get('phase', '-')}")
        metrics = data.get("totals", data.get("metrics", {}))
        progress = train.get("progress", {})
        current = (evaluation.get("result", {}).get("publication") or state.get("current") or {})
        lines = [f"模式 {state.get('mode', 'teacher')}，训练来源 {state.get('training_sources')}，权重 {state.get('source_weights')}",
            f"采集局数 {state['games']}，候选轮数 {len(state['rounds'])}",
            f"池累计生成 {metrics.get('generated', 0)}，有效 {metrics.get('valid', 0)}，入池 {metrics.get('inserted', 0)}",
            f"池容量 {metrics.get('total_pool_samples', metrics.get('pool_samples', 0))}/{metrics.get('capacity', '?')}，可用 {metrics.get('pool_samples', 0)}，有效/秒 {metrics.get('effective_per_second', 0):.2f}",
            f"采集模型 {data.get('version') or data.get('model_id', 'handcrafted')}",
            f"训练 step {progress.get('step', '-')}，loss {progress.get('train_loss', '-')}，val MAE {progress.get('val_mae', '-')}",
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
            if kind != "error":
                self.show_state(state)
            if kind == "state":
                continue
            self.thread = None
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self._enable_settings(True)
            self._mode_changed()
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
