"""
System tray UI component for QuickScribe.

Provides visual feedback of application state through system tray icon
and quick access to controls via context menu.
"""

from enum import Enum
from typing import Optional
from PyQt6.QtWidgets import QSystemTrayIcon, QMenu
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtCore import QObject, pyqtSignal
from .dictation_history_window import DictationHistoryWindow


class AppState(Enum):
    """Application states reflected in system tray."""
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    ERROR = "error"


class SystemTrayUI(QObject):
    """
    System tray icon that reflects application state.

    Signals:
        start_recording_requested: User clicked start recording
        stop_recording_requested: User clicked stop recording
        quit_requested: User clicked quit
        window_focus_requested: User clicked window focus notification, emits window ID
    """

    start_recording_requested = pyqtSignal()
    stop_recording_requested = pyqtSignal()
    quit_requested = pyqtSignal()
    window_focus_requested = pyqtSignal(str)

    # Internal signals for cross-thread marshaling.
    # Worker threads emit these; Qt auto-connection delivers the connected
    # slots on the main thread that owns this QObject.
    _request_set_state = pyqtSignal(object)
    _request_show_message = pyqtSignal(str, str)
    _request_show_error = pyqtSignal(str)
    _request_window_focus = pyqtSignal(str, str)
    _request_add_history = pyqtSignal(str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)

        self._current_state = AppState.IDLE
        self._tray_icon = QSystemTrayIcon(self)
        self._menu = QMenu()
        self._pending_window_id: Optional[str] = None
        self._history_window = DictationHistoryWindow()

        self._setup_menu()
        self._setup_tray()

        self._request_set_state.connect(self._do_set_state)
        self._request_show_message.connect(self._do_show_message)
        self._request_show_error.connect(self._do_show_error)
        self._request_window_focus.connect(self._do_window_focus)
        self._request_add_history.connect(self._history_window.add_entry)

    def _setup_menu(self):
        """Create context menu for tray icon."""
        self._action_start = QAction("Start Recording", self)
        self._action_start.triggered.connect(self.start_recording_requested.emit)
        self._menu.addAction(self._action_start)

        self._action_stop = QAction("Stop Recording", self)
        self._action_stop.triggered.connect(self.stop_recording_requested.emit)
        self._action_stop.setEnabled(False)
        self._menu.addAction(self._action_stop)

        self._menu.addSeparator()

        self._action_history = QAction("Dictation History", self)
        self._action_history.triggered.connect(self._toggle_history_window)
        self._menu.addAction(self._action_history)

        self._menu.addSeparator()

        self._action_quit = QAction("Quit", self)
        self._action_quit.triggered.connect(self.quit_requested.emit)
        self._menu.addAction(self._action_quit)

    def _setup_tray(self):
        """Initialize tray icon."""
        self._tray_icon.setContextMenu(self._menu)
        self._tray_icon.messageClicked.connect(self._on_focus_notification_clicked)
        self._update_icon()
        self._tray_icon.show()

    def _update_icon(self):
        """Update icon based on current state."""
        import os

        # Get icon directory path relative to this file
        icon_dir = os.path.join(os.path.dirname(__file__), 'icons')

        if self._current_state == AppState.RECORDING:
            icon_path = os.path.join(icon_dir, 'recording.svg')
            tooltip = "QuickScribe - Recording"
        elif self._current_state == AppState.PROCESSING:
            icon_path = os.path.join(icon_dir, 'processing.svg')
            tooltip = "QuickScribe - Processing"
        elif self._current_state == AppState.ERROR:
            icon_path = os.path.join(icon_dir, 'error.svg')
            tooltip = "QuickScribe - Error"
        else:  # IDLE
            icon_path = os.path.join(icon_dir, 'idle.svg')
            tooltip = "QuickScribe - Idle"

        icon = QIcon(icon_path)
        self._tray_icon.setIcon(icon)
        self._tray_icon.setToolTip(tooltip)

    def set_state(self, state: AppState):
        """
        Update tray icon to reflect new application state.

        Thread-safe: marshals to main thread via signal.
        """
        self._request_set_state.emit(state)

    def show_message(self, title: str, message: str):
        """
        Show notification message from tray icon.

        Thread-safe: marshals to main thread via signal.
        """
        self._request_show_message.emit(title, message)

    def show_error(self, error_message: str):
        """
        Display error state and show toast notification.

        Thread-safe: marshals to main thread via signal.
        """
        self._request_show_error.emit(error_message)

    def show_window_focus_notification(self, window_id: Optional[str], message: str):
        """
        Display clickable notification to focus specific window.

        Thread-safe: marshals to main thread via signal.
        """
        if window_id is None:
            return

        self._request_window_focus.emit(window_id, message)

    def _do_set_state(self, state: AppState):
        """Main-thread slot: update icon and menu actions for new state."""
        self._current_state = state
        self._update_icon()

        if state == AppState.RECORDING:
            self._action_start.setEnabled(False)
            self._action_stop.setEnabled(True)
        else:
            self._action_start.setEnabled(True)
            self._action_stop.setEnabled(False)

    def _do_show_message(self, title: str, message: str):
        """Main-thread slot: display tray notification toast."""
        self._tray_icon.showMessage(title, message)

    def _do_show_error(self, error_message: str):
        """Main-thread slot: set error state and display critical toast."""
        self._do_set_state(AppState.ERROR)
        self._tray_icon.showMessage("Dictation API error", error_message, QSystemTrayIcon.MessageIcon.Critical, 3000)

    def _do_window_focus(self, window_id: str, message: str):
        """Main-thread slot: display persistent clickable notification for window focus."""
        self._pending_window_id = window_id
        self._tray_icon.showMessage("Window Focus Required", message, QSystemTrayIcon.MessageIcon.Information, 0)

    def _on_focus_notification_clicked(self):
        """Handle click on window focus notification."""
        if self._pending_window_id is None:
            return

        window_id = self._pending_window_id
        self._pending_window_id = None
        self.window_focus_requested.emit(window_id)

    def add_dictation_history(self, text: str):
        """
        Append dictation text to the history window.

        Thread-safe: marshals to main thread via signal.
        """
        self._request_add_history.emit(text)

    def _toggle_history_window(self):
        """Show or raise the dictation history window."""
        if not self._history_window.isVisible():
            self._history_window.show()

        self._history_window.raise_()
        self._history_window.activateWindow()

    def cleanup(self):
        """Clean up tray icon resources."""
        self._history_window.close()
        self._tray_icon.hide()
