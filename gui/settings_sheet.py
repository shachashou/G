"""Right-side settings sheet with a dimmed backdrop.

All advanced controls are moved here from the main cards, but the widgets and
their bindings stay unchanged. The sheet is non-destructive: background tasks
keep running while it is open.
"""

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from ui_components import mark_button


class SettingsSheet(QObject):
    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.parent = parent
        self.scrim = QFrame(parent)
        self.scrim.setObjectName("Scrim")
        self.scrim.hide()
        self.scrim.installEventFilter(self)
        self.frame = QFrame(parent)
        self.frame.setObjectName("Sheet")
        self.frame.setFixedWidth(520)
        self.frame.hide()
        outer = QVBoxLayout(self.frame)
        outer.setContentsMargins(22, 20, 22, 18)
        outer.setSpacing(12)
        header = QHBoxLayout()
        title = QLabel("设置")
        title.setObjectName("SheetTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.btn_close = QPushButton("关闭")
        self.btn_close.setObjectName("Quiet")
        self.btn_close.setFixedWidth(48)
        self.btn_close.clicked.connect(self.hide)
        header.addWidget(self.btn_close)
        outer.addLayout(header)
        hint = QLabel("高级参数只影响新任务；任务运行时暂停/恢复仍使用训练页的主操作。")
        hint.setObjectName("Caption")
        hint.setWordWrap(True)
        outer.addWidget(hint)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        self.content = QVBoxLayout(content)
        self.content.setContentsMargins(0, 0, 4, 0)
        self.content.setSpacing(14)
        self.scroll.setWidget(content)
        outer.addWidget(self.scroll, 1)
        self._sections: dict[str, QWidget] = {}
        self._hidden = True

    def add_section(self, key: str, title: str, widget: QWidget) -> None:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        caption = QLabel(title)
        caption.setObjectName("SheetSection")
        layout.addWidget(caption)
        layout.addWidget(widget)
        self.content.addWidget(box)
        self._sections.setdefault(key, box)

    def show(self, section: str | None = None) -> None:
        self.reposition()
        self.scrim.show()
        self.scrim.raise_()
        self.frame.show()
        self.frame.raise_()
        self._hidden = False
        if section and section in self._sections:
            self.scroll.ensureWidgetVisible(self._sections[section], 0, 8)

    def hide(self) -> None:
        self.frame.hide()
        self.scrim.hide()
        self._hidden = True
        if hasattr(self.parent, "btn_settings"):
            self.parent.btn_settings.setFocus()

    def visible(self) -> bool:
        return not self._hidden

    def reposition(self) -> None:
        rect = self.parent.rect()
        width = min(520, max(340, rect.width() - 40))
        self.frame.setFixedWidth(width)
        self.frame.setGeometry(rect.width() - width - 16, 14,
                               width, max(200, rect.height() - 28))
        self.scrim.setGeometry(rect)

    def eventFilter(self, watched, event) -> bool:
        if watched is self.scrim and event.type() == QEvent.Type.MouseButtonPress:
            self.hide()
            return True
        return False


def sheet_button(text: str, role: str = "quiet") -> QPushButton:
    button = QPushButton(text)
    mark_button(button, role)
    return button
