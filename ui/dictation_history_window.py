"""
Dictation history window showing recent dictation results.

Displays the last 100 dictations in a scrollable text area,
with newest entries at the bottom and auto-scroll.
History is persisted to disk and survives application restarts.
"""

import json
import os
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPlainTextEdit, QCheckBox
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QTextCursor

from lib.app_dirs import ensure_data_dir, ensure_config_dir

HISTORY_MAX_ENTRIES = 100
_HISTORY_FILENAME = "dictation_history.json"
_SETTINGS_FILENAME = "history_settings.json"


@dataclass
class DictationRecord:
    """A single completed dictation with metadata."""
    timestamp: str
    text: str


def _load_history(path: str) -> deque:
    """Load persisted history records from JSON file."""
    entries: deque = deque(maxlen=HISTORY_MAX_ENTRIES)
    if not os.path.exists(path):
        return entries

    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        for item in raw:
            entries.append(DictationRecord(**item))
    except (OSError, json.JSONDecodeError, TypeError, KeyError):
        # Silent return: corrupt or unreadable history is non-critical;
        # app starts with empty history rather than crashing.
        pass

    return entries


def _save_history(path: str, entries: deque):
    """Persist history records to JSON file."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in entries], f, indent=2, ensure_ascii=False)
    except OSError:
        # Silent return: history persistence is best-effort; loss of a save
        # does not affect in-memory state or dictation functionality.
        pass


def _load_settings(path: str) -> dict:
    """Load UI settings from JSON file."""
    if not os.path.exists(path):
        return {}

    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        # Silent return: corrupt settings file falls back to defaults;
        # UI preference loss is non-critical.
        return {}


def _save_settings(path: str, settings: dict):
    """Persist UI settings to JSON file."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except OSError:
        # Silent return: settings persistence is best-effort; the toggle
        # still functions in-session if the write fails.
        pass


class DictationHistoryWindow(QWidget):
    """Window displaying recent dictation history with optional timestamps."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("QuickScribe - Dictation History")
        self.setMinimumSize(500, 400)
        self.resize(600, 500)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
        )

        self._history_path = os.path.join(ensure_data_dir(), _HISTORY_FILENAME)
        self._settings_path = os.path.join(ensure_config_dir(), _SETTINGS_FILENAME)

        settings = _load_settings(self._settings_path)
        self._show_timestamps: bool = settings.get("show_timestamps", False)

        self._entries: deque = _load_history(self._history_path)

        self._build_ui()
        self._rebuild_display()

    def _build_ui(self):
        """Construct the window layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)

        self._timestamp_toggle = QCheckBox("Show timestamps", self)
        self._timestamp_toggle.setChecked(self._show_timestamps)
        self._timestamp_toggle.stateChanged.connect(self._on_timestamp_toggle)
        toolbar.addWidget(self._timestamp_toggle)
        toolbar.addStretch()

        layout.addLayout(toolbar)

        self._text_area = QPlainTextEdit(self)
        self._text_area.setReadOnly(True)
        self._text_area.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        layout.addWidget(self._text_area)

    @pyqtSlot(str)
    def add_entry(self, text: str):
        """
        Append a dictation record with timestamp.

        Rebuilds display when entry count exceeds maximum to reflect
        eviction from the deque.
        """
        record = DictationRecord(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            text=text,
        )
        was_full = len(self._entries) == HISTORY_MAX_ENTRIES
        self._entries.append(record)
        _save_history(self._history_path, self._entries)

        if was_full:
            self._rebuild_display()
        else:
            self._append_record(record)

    def _format_record(self, record: DictationRecord) -> str:
        """Format a record for display according to current timestamp setting."""
        if self._show_timestamps:
            return f"[{record.timestamp}] {record.text}"
        return record.text

    def _append_record(self, record: DictationRecord):
        """Append a single formatted record and scroll to bottom."""
        self._text_area.appendPlainText(self._format_record(record))
        self._scroll_to_bottom()

    def _rebuild_display(self):
        """Rebuild entire text area from records."""
        self._text_area.setPlainText(
            "\n".join(self._format_record(r) for r in self._entries)
        )
        self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        """Move cursor and scrollbar to the end."""
        cursor = self._text_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._text_area.setTextCursor(cursor)
        scrollbar = self._text_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_timestamp_toggle(self, state: int):
        """Handle timestamp checkbox state change."""
        self._show_timestamps = state == Qt.CheckState.Checked.value
        _save_settings(self._settings_path, {"show_timestamps": self._show_timestamps})
        self._rebuild_display()

    def closeEvent(self, event):
        """Hide instead of destroying so state is preserved."""
        event.ignore()
        self.hide()
