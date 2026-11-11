"""Small visual vocabulary for the simplified desktop workbench.

The UI deliberately follows a phone-app hierarchy: one calm surface, white
cards, one blue primary action per mode, and progressive disclosure for the
many training/evaluation knobs. No rules or task behavior live here.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget,
)

APP_STYLE = """
QWidget {
    font-family: "Segoe UI Variable", "SF Pro Text", "Segoe UI", "Microsoft YaHei UI";
    font-size: 12px;
    color: #1D1D1F;
}
QMainWindow, QWidget#Root { background: #F5F5F7; }
QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; border: none; }
QScrollArea { border: none; }
QFrame#BoardCard {
    background: #FFFFFF;
    border: 1px solid #EDEDF2;
    border-radius: 20px;
}
QGroupBox {
    background: #FFFFFF;
    border: 1px solid #EDEDF2;
    border-radius: 16px;
    margin-top: 10px;
    padding: 14px 12px 12px 12px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 0 6px;
    color: #6E6E73;
    font-size: 10px;
    font-weight: 600;
}
QLabel#Title { font-size: 24px; font-weight: 700; letter-spacing: -0.5px; }
QLabel#Subtitle { font-size: 12px; color: #6E6E73; }
QLabel#Pill {
    background: #FFFFFF;
    border: 1px solid #E5E5EA;
    border-radius: 12px;
    padding: 5px 10px;
    font-size: 11px;
    color: #6E6E73;
}
QLabel#Caption { color: #6E6E73; font-size: 11px; }
QPushButton {
    background: #FFFFFF;
    border: 1px solid #D2D2D7;
    border-radius: 10px;
    padding: 6px 10px;
    min-height: 18px;
    font-size: 12px;
    font-weight: 600;
    color: #007AFF;
}
QPushButton:hover { background: #F2F2F7; }
QPushButton:pressed { background: #E8E8ED; }
QPushButton:disabled { color: #AEAEB2; background: #F5F5F7; border-color: #EDEDF2; }
QPushButton#Primary { background: #007AFF; border: 1px solid #007AFF; color: #FFFFFF; }
QPushButton#Primary:hover { background: #0A84FF; border-color: #0A84FF; }
QPushButton#Primary:disabled { background: #B7D7FF; border-color: #B7D7FF; color: #FFFFFF; }
QPushButton#Danger { color: #FF3B30; }
QPushButton#Quiet { border: none; background: transparent; padding: 5px 8px; }
QPushButton#Quiet:hover { background: #F2F2F7; }
QPushButton#Disclosure {
    border: none;
    background: transparent;
    color: #6E6E73;
    text-align: left;
    padding: 5px 2px;
    font-size: 11px;
    font-weight: 600;
}
QPushButton#Disclosure:hover { color: #1D1D1F; background: transparent; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #FFFFFF;
    border: 1px solid #D2D2D7;
    border-radius: 8px;
    padding: 5px 7px;
    min-height: 16px;
    selection-background-color: #007AFF;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #007AFF;
}
QComboBox::drop-down { border: none; width: 22px; }
QComboBox#ModelCapsule { border-radius: 14px; padding: 6px 10px; }
QComboBox#ModelCapsule:focus { border: 1px solid #D2D2D7; }
QComboBox QAbstractItemView {
    background: #FFFFFF;
    border: 1px solid #E5E5EA;
    border-radius: 10px;
    selection-background-color: #E8F0FF;
    selection-color: #1D1D1F;
}
QTextEdit, QListWidget {
    background: #FFFFFF;
    border: 1px solid #E5E5EA;
    border-radius: 12px;
    padding: 6px;
}
QCheckBox, QRadioButton { spacing: 8px; }
QTabWidget::pane { border: none; background: transparent; top: -1px; }
QTabBar::tab {
    background: transparent;
    color: #6E6E73;
    padding: 6px 10px;
    margin: 0 2px;
    border-radius: 9px;
    font-weight: 600;
}
QTabBar::tab:selected { background: #FFFFFF; color: #1D1D1F; }
QTabBar::tab:hover { color: #1D1D1F; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 0; }
QScrollBar::handle:vertical { background: #C7C7CC; border-radius: 4px; min-height: 28px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: transparent; height: 8px; }
QScrollBar::handle:horizontal { background: #C7C7CC; border-radius: 4px; min-width: 28px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QToolTip { background: #1D1D1F; color: #FFFFFF; border: none; padding: 6px 8px; }
QFrame#Scrim { background-color: rgba(0, 0, 0, 88); }
QFrame#Sheet {
    background: #F5F5F7;
    border: 1px solid #E5E5EA;
    border-radius: 20px;
}
QLabel#SheetTitle { font-size: 20px; font-weight: 700; letter-spacing: -0.4px; }
QLabel#SheetSection { color: #6E6E73; font-size: 10px; font-weight: 600; }
QPushButton#Segment {
    border: 1px solid transparent;
    background: transparent;
    color: #6E6E73;
    border-radius: 9px;
    padding: 5px 10px;
}
QPushButton#Segment:checked {
    background: #FFFFFF;
    color: #1D1D1F;
    border-color: #E5E5EA;
}
QPushButton#Segment:hover { color: #1D1D1F; }
QPushButton#Live {
    background: #FF3B30;
    border: 1px solid #FF3B30;
    color: #FFFFFF;
    border-radius: 11px;
    padding: 4px 10px;
    font-size: 10px;
    font-weight: 700;
}
QPushButton#Live:hover { background: #FF453A; }
QComboBox#ModelCapsule:focus { border: 1px solid #D2D2D7; }
"""


def apply_theme(app) -> None:
    app.setStyleSheet(APP_STYLE)


def mark_button(button: QPushButton, role: str) -> QPushButton:
    """role: primary | danger | quiet | default (secondary)."""
    button.setObjectName({
        "primary": "Primary", "danger": "Danger", "quiet": "Quiet",
    }.get(role, ""))
    return button


def page_title(title: str, subtitle: str = "") -> QWidget:
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(2, 0, 2, 6)
    layout.setSpacing(2)
    heading = QLabel(title)
    heading.setObjectName("Title")
    layout.addWidget(heading)
    if subtitle:
        caption = QLabel(subtitle)
        caption.setObjectName("Subtitle")
        caption.setWordWrap(False)
        layout.addWidget(caption)
    return widget


class SegmentedControl(QWidget):
    """A compact iOS-like three/four-way switch."""

    changed = Signal(str)

    def __init__(self, items: list[tuple[str, str]], current: str | None = None):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: dict[str, QPushButton] = {}
        for key, label in items:
            button = QPushButton(label)
            button.setCheckable(True)
            button.setObjectName("Segment")
            button.setCursor(Qt.PointingHandCursor)
            self.group.addButton(button)
            layout.addWidget(button)
            self.buttons[key] = button
        self.group.buttonClicked.connect(self._clicked)
        self._current = ""
        self.set_current(current or items[0][0])

    def _clicked(self, button) -> None:
        for key, candidate in self.buttons.items():
            if candidate is button:
                self._current = key
                self.changed.emit(key)
                return

    def set_current(self, key: str) -> None:
        button = self.buttons.get(key)
        if button is None:
            return
        button.setChecked(True)
        self._current = key

    def current(self) -> str:
        return self._current


class CollapsibleSection(QWidget):
    """A quiet disclosure row; collapsed by default to keep the main view calm."""

    def __init__(self, title: str, layout, expanded: bool = False) -> None:
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Maximum)
        self.expanded = False
        self._title = title
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        self.toggle = QPushButton()
        self.toggle.setObjectName("Disclosure")
        self.toggle.setCursor(Qt.PointingHandCursor)
        self.toggle.clicked.connect(self.toggle_expanded)
        outer.addWidget(self.toggle)
        self.body = QWidget()
        self.body.setLayout(layout)
        outer.addWidget(self.body)
        self.set_expanded(expanded)

    def set_expanded(self, expanded: bool) -> None:
        self.expanded = bool(expanded)
        self.body.setVisible(self.expanded)
        arrow = "⌄" if self.expanded else "›"
        self.toggle.setText(f"{arrow}  {self._title}")

    def toggle_expanded(self) -> None:
        self.set_expanded(not self.expanded)

    def embed(self) -> None:
        """Show the content permanently, e.g. inside the settings sheet."""
        self.toggle.hide()
        self.set_expanded(True)
        self.body.setVisible(True)


def card_frame(layout) -> QFrame:
    frame = QFrame()
    frame.setObjectName("Card")
    frame.setLayout(layout)
    return frame


def row(*widgets, spacing: int = 8) -> QHBoxLayout:
    layout = QHBoxLayout()
    layout.setSpacing(spacing)
    for widget in widgets:
        if widget is None:
            layout.addStretch(1)
        elif isinstance(widget, int):
            layout.addSpacing(widget)
        else:
            layout.addWidget(widget)
    return layout
