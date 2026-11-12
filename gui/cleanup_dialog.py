"""Dialog for removing candidate models and stale historical data.

Two tabs share one guarded deletion flow:

* 候选模型 — candidate bundles from ``tools/model_cleanup.py``
* 历史数据 — finished task directories and old evaluation runs from
  ``tools/storage_cleanup.py``

The dialog only *offers* selections; both planners re-scan and re-check every
protection before touching the disk.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSpinBox,
    QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

import model_cleanup
import storage_cleanup
from model_cleanup import human_size
from ui_components import mark_button


class CleanupDialog(QDialog):
    cleaned = Signal(int, int)  # deleted count, freed bytes

    def __init__(self, weights_root: str | Path, registry_root: str | Path,
                 evaluations_root: str | Path, loaded_model: str | None = None,
                 keep: int = 1, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("清理模型与历史数据")
        self.resize(820, 600)
        self._weights = Path(weights_root)
        self._registry = Path(registry_root)
        self._evaluations = Path(evaluations_root)
        self._loaded = loaded_model
        self.keep = max(0, int(keep))
        self._bundles: list[model_cleanup.Bundle] = []
        self._data: list[storage_cleanup.DataEntry] = []
        self._build_ui()
        self.reload()

    # ---- layout ------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 18, 20, 16)
        outer.setSpacing(10)

        intro = QLabel(
            "删除不再需要的候选模型和训练历史。正式/历史版本证据、当前加载的"
            "模型、测试样例和运行中的任务永远保留；可恢复任务默认不勾选。")
        intro.setWordWrap(True)
        intro.setObjectName("Caption")
        outer.addWidget(intro)

        self.tabs = QTabWidget()

        # ---- candidate models ---------------------------------------
        model_page = _Page()
        model_body = QLabel(
            "只删除候选模型包（模型 + 检查点 + 训练位置）。每个任务至少保留"
            "最近几轮，所以暂停的任务仍能恢复。")
        model_body.setWordWrap(True)
        model_body.setObjectName("Caption")
        model_page.layout.addWidget(model_body)
        keep_row = QHBoxLayout()
        keep_row.addWidget(QLabel("每个任务保留最近"))
        self.spin_keep = QSpinBox()
        self.spin_keep.setRange(0, 20)
        self.spin_keep.setValue(self.keep)
        self.spin_keep.valueChanged.connect(self._keep_changed)
        keep_row.addWidget(self.spin_keep)
        keep_row.addWidget(QLabel("轮候选"))
        keep_row.addStretch(1)
        model_page.layout.addLayout(keep_row)
        self.model_tree = self._make_tree(("模型", "大小", "说明"))
        model_page.layout.addWidget(self.model_tree, 1)
        self.model_summary = QLabel("")
        self.model_summary.setObjectName("Caption")
        self.model_summary.setWordWrap(True)
        model_page.layout.addWidget(self.model_summary)
        self.tabs.addTab(model_page.widget, "候选模型")

        # ---- historical data ----------------------------------------
        data_page = _Page()
        data_body = QLabel(
            "已完成/失败的任务不在恢复列表里，可以删除；可恢复任务（暂停/已停止）"
            "删除后无法继续。测评记录只有被登记版本引用时才保留证据。")
        data_body.setWordWrap(True)
        data_body.setObjectName("Caption")
        data_page.layout.addWidget(data_body)
        self.data_tree = self._make_tree(("任务 / 测评", "大小", "说明"))
        data_page.layout.addWidget(self.data_tree, 1)
        self.data_summary = QLabel("")
        self.data_summary.setObjectName("Caption")
        self.data_summary.setWordWrap(True)
        data_page.layout.addWidget(self.data_summary)
        self.tabs.addTab(data_page.widget, "历史数据")

        outer.addWidget(self.tabs, 1)

        footer = QHBoxLayout()
        self.summary_label = QLabel("")
        self.summary_label.setObjectName("Caption")
        footer.addWidget(self.summary_label, 1)
        self.btn_select = QPushButton("全选可清理")
        mark_button(self.btn_select, "quiet")
        self.btn_select.clicked.connect(self._select_all)
        footer.addWidget(self.btn_select)
        self.btn_close = QPushButton("关闭")
        mark_button(self.btn_close, "quiet")
        self.btn_close.clicked.connect(self.reject)
        footer.addWidget(self.btn_close)
        self.btn_delete = QPushButton("删除选中")
        mark_button(self.btn_delete, "danger")
        self.btn_delete.clicked.connect(self._delete)
        footer.addWidget(self.btn_delete)
        outer.addLayout(footer)

    def _make_tree(self, headers: tuple[str, str, str]) -> QTreeWidget:
        tree = QTreeWidget()
        tree.setColumnCount(3)
        tree.setHeaderLabels(list(headers))
        tree.setUniformRowHeights(True)
        tree.setSelectionMode(QTreeWidget.SelectionMode.NoSelection)
        tree.setColumnWidth(0, 400)
        tree.setColumnWidth(1, 90)
        tree.itemChanged.connect(self._sync_summary)
        return tree

    # ---- data --------------------------------------------------------
    def reload(self) -> None:
        self._reload_models()
        self._reload_data()
        self._sync_summary()

    def _reload_models(self) -> None:
        try:
            self._bundles = model_cleanup.scan(
                self._weights, registry_root=self._registry,
                loaded_model=self._loaded, keep=self.keep)
        except ValueError as exc:
            self._bundles = []
            self.model_summary.setText(str(exc))
        rows = [{
            "id": bundle.id,
            "group": bundle.group,
            "group_label": bundle.group_label,
            "label": (bundle.id.split("/")[-1] if bundle.kind == "task"
                      else bundle.id),
            "size": bundle.size,
            "protection": bundle.protection,
            "warning": bundle.warning,
            "checked": bundle.kind == "task",
            "tooltip": str(bundle.directory),
        } for bundle in self._bundles]
        self._populate(self.model_tree, rows)
        data = model_cleanup.summary(self._bundles)
        self.model_summary.setText(
            f"候选 {data['total']} 个 · {human_size(data['total_bytes'])}；"
            f"可清理 {data['removable']} 个 · "
            f"{human_size(data['removable_bytes'])}"
            f"（默认勾选 {data['default_selected']} 个）")

    def _reload_data(self) -> None:
        try:
            self._data = storage_cleanup.scan(
                self._weights, self._evaluations,
                registry_root=self._registry)
        except ValueError as exc:
            self._data = []
            self.data_summary.setText(str(exc))
        rows = [{
            "id": entry.id,
            "group": entry.group,
            "group_label": entry.group_label,
            "label": entry.label,
            "size": entry.size,
            "protection": entry.protection,
            "warning": entry.warning,
            "checked": entry.default_checked,
            "tooltip": str(entry.path),
        } for entry in self._data]
        self._populate(self.data_tree, rows)
        data = storage_cleanup.summary(self._data)
        self.data_summary.setText(
            f"历史数据 {data['total']} 项 · {human_size(data['total_bytes'])}；"
            f"可清理 {data['removable']} 项 · "
            f"{human_size(data['removable_bytes'])}"
            f"（默认勾选 {data['default_selected']} 项）")

    def _populate(self, tree: QTreeWidget, rows: list[dict]) -> None:
        tree.blockSignals(True)
        tree.clear()
        groups: dict[str, QTreeWidgetItem] = {}
        for row in rows:
            parent = groups.get(row["group"])
            if parent is None:
                parent = QTreeWidgetItem(tree, [row["group_label"], "", ""])
                font = parent.font(0)
                font.setBold(True)
                parent.setFont(0, font)
                parent.setFlags(Qt.ItemIsEnabled)
                groups[row["group"]] = parent
            item = QTreeWidgetItem(parent, [
                row["label"], human_size(row["size"]),
                self._note(row["protection"], row["warning"])])
            item.setData(0, Qt.UserRole, row["id"])
            item.setToolTip(0, row["tooltip"])
            if row["protection"] is None:
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(0, Qt.Checked if row["checked"]
                                   else Qt.Unchecked)
            else:
                item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable)
                item.setCheckState(0, Qt.Unchecked)
                item.setForeground(0, QBrush(Qt.GlobalColor.gray))
        tree.expandAll()
        tree.blockSignals(False)

    @staticmethod
    def _note(protection: str | None, warning: str | None) -> str:
        if protection:
            return "🔒 " + protection
        if warning:
            return "⚠ " + warning
        return "可清理"

    def _keep_changed(self, value: int) -> None:
        self.keep = int(value)
        self._reload_models()
        self._sync_summary()

    # ---- selection ---------------------------------------------------
    def _selected(self, tree: QTreeWidget) -> list[str]:
        ids: list[str] = []
        for i in range(tree.topLevelItemCount()):
            group = tree.topLevelItem(i)
            for j in range(group.childCount()):
                item = group.child(j)
                if item.checkState(0) == Qt.Checked:
                    ids.append(str(item.data(0, Qt.UserRole)))
        return ids

    def _selected_size(self, tree: QTreeWidget, entries) -> int:
        wanted = set(self._selected(tree))
        return sum(entry.size for entry in entries if entry.id in wanted)

    def _sync_summary(self, *_args) -> None:
        model_ids = self._selected(self.model_tree)
        data_ids = self._selected(self.data_tree)
        pieces = []
        if model_ids:
            pieces.append(f"候选 {len(model_ids)} 个 / "
                          f"{human_size(self._selected_size(self.model_tree, self._bundles))}")
        if data_ids:
            pieces.append(f"历史数据 {len(data_ids)} 项 / "
                          f"{human_size(self._selected_size(self.data_tree, self._data))}")
        self.summary_label.setText(
            ("已选 " + "，".join(pieces)) if pieces
            else "未选择要删除的内容")
        self.btn_delete.setEnabled(bool(model_ids or data_ids))
        self.btn_select.setEnabled(any(
            entry.removable for entry in (*self._bundles, *self._data)))

    def _select_all(self) -> None:
        for tree, entries in ((self.model_tree, self._bundles),
                              (self.data_tree, self._data)):
            tree.blockSignals(True)
            for i in range(tree.topLevelItemCount()):
                group = tree.topLevelItem(i)
                for j in range(group.childCount()):
                    item = group.child(j)
                    if item.flags() & Qt.ItemIsUserCheckable:
                        item.setCheckState(0, Qt.Checked)
            tree.blockSignals(False)
        self._sync_summary()

    # ---- deletion ----------------------------------------------------
    def _delete(self) -> None:
        model_ids = self._selected(self.model_tree)
        data_ids = self._selected(self.data_tree)
        if not model_ids and not data_ids:
            return
        total = (self._selected_size(self.model_tree, self._bundles)
                 + self._selected_size(self.data_tree, self._data))
        what = []
        if model_ids:
            what.append(f"{len(model_ids)} 个候选模型包")
        if data_ids:
            what.append(f"{len(data_ids)} 项历史数据")
        answer = QMessageBox.question(
            self, "清理",
            "将删除 " + "、".join(what) + f"（约 {human_size(total)}）。\n"
            "正式/历史版本证据、当前加载模型和运行中任务不受影响；"
            "删除后无法恢复。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        results: list = []
        if model_ids:
            results.extend(model_cleanup.delete(
                self._weights, model_ids, registry_root=self._registry,
                loaded_model=self._loaded, keep=self.keep))
        if data_ids:
            results.extend(storage_cleanup.delete(
                self._weights, self._evaluations, data_ids,
                registry_root=self._registry))
        failed = [result for result in results if not result.ok]
        freed = sum(result.freed for result in results if result.ok)
        deleted = len(results) - len(failed)
        self.reload()
        self.cleaned.emit(deleted, freed)
        if failed:
            reasons = "\n".join(
                f"{result.id}：{result.message}" for result in failed[:5])
            QMessageBox.warning(self, "清理",
                                f"已删除 {deleted} 项，{len(failed)} 项被跳过：\n{reasons}")
        else:
            QMessageBox.information(
                self, "清理",
                f"已删除 {deleted} 项，释放 {human_size(freed)}。")


class _Page:
    """Small helper: a QWidget with a vertical layout ready to fill."""

    def __init__(self) -> None:
        self.widget = QWidget()
        self.layout = QVBoxLayout(self.widget)
        self.layout.setContentsMargins(0, 8, 0, 0)
        self.layout.setSpacing(8)
