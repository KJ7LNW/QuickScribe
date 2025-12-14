"""Xdotool implementation of KeyboardInjector interface."""

import subprocess
import os
import sys
import time
import threading
from typing import Optional, TYPE_CHECKING
sys.path.append(os.path.join(os.path.dirname(__file__), 'xml-stream'))
from keyboard_injector import KeyboardInjector
sys.path.insert(0, os.path.dirname(__file__))
from pr_log import pr_err, pr_debug
from pynput import keyboard

if TYPE_CHECKING:
    from processing_session import ProcessingSession


class ModifierStateTracker:
    """Tracks modifier key states using pynput keyboard listener."""

    def __init__(self):
        self._modifiers = {
            'ctrl': False,
            'alt': False,
            'shift': False,
            'super': False
        }
        self._lock = threading.Lock()
        self._no_modifiers_event = threading.Event()
        self._no_modifiers_event.set()

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self._listener.start()

    def _on_press(self, key):
        with self._lock:
            if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                self._modifiers['ctrl'] = True
            elif key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r):
                self._modifiers['alt'] = True
            elif key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
                self._modifiers['shift'] = True
            elif key in (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r):
                self._modifiers['super'] = True

            if any(self._modifiers.values()):
                self._no_modifiers_event.clear()

    def _on_release(self, key):
        with self._lock:
            if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                self._modifiers['ctrl'] = False
            elif key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r):
                self._modifiers['alt'] = False
            elif key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
                self._modifiers['shift'] = False
            elif key in (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r):
                self._modifiers['super'] = False

            if not any(self._modifiers.values()):
                self._no_modifiers_event.set()

    def wait_for_no_modifiers(self) -> None:
        """Block until all modifier keys are released, then wait additional 100ms for propagation."""
        had_to_wait = not self._no_modifiers_event.is_set()
        self._no_modifiers_event.wait()
        if had_to_wait:
            time.sleep(0.1)

    def stop(self) -> None:
        """Stop the keyboard listener."""
        self._listener.stop()


class XdotoolKeyboardInjector(KeyboardInjector):
    """Xdotool-based keyboard injector for direct system keyboard operations."""

    # Stabilization delay after window activation to prevent character clipping
    # Allows window manager and target application to fully process activation before typing begins
    WINDOW_ACTIVATION_STABILIZATION_DELAY = 0.5

    def __init__(self, config=None, typing_delay: int = 5):
        """
        Initialize xdotool keyboard injector.

        Args:
            config: Configuration object with xdotool_rate and debug_enabled
            typing_delay: Default millisecond delay between keystrokes if no config
        """
        xdotool_rate = getattr(config, 'xdotool_rate', None) if config else None
        if xdotool_rate:
            # Convert Hz to milliseconds delay: delay = 1000 / rate
            self.typing_delay = int(1000 / xdotool_rate)
            if getattr(config, 'debug_enabled', False):
                pr_debug(f"XdotoolKeyboardInjector: typing_rate={xdotool_rate}Hz -> delay={self.typing_delay}ms")
        else:
            self.typing_delay = typing_delay
            if config and getattr(config, 'debug_enabled', False):
                pr_debug(f"XdotoolKeyboardInjector: using default typing_delay={self.typing_delay}ms")
        self.debug_enabled = getattr(config, 'debug_enabled', False) if config else False
        self.test_mode = (
            os.getenv("TESTING", "false").lower() == "true" or
            "pytest" in os.getenv("_", "") or
            "pytest" in str(os.getenv("PYTEST_CURRENT_TEST", "")) or
            any("pytest" in arg for arg in sys.argv if arg)
        )
        self._modifier_tracker = ModifierStateTracker()
        self._session_window_id: Optional[str] = None
        self._session_window_activated_event: Optional[threading.Event] = None

    def get_trigger_window_id(self) -> Optional[str]:
        """Get active window ID at trigger time using xdotool."""
        if self.test_mode:
            return None

        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow"],
                check=True,
                capture_output=True,
                text=True
            )
            window_id = result.stdout.strip()
            if self.debug_enabled:
                pr_debug(f"Captured trigger window ID: {window_id}")
            return window_id
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            if self.debug_enabled:
                pr_debug(f"Failed to capture window ID: {str(e)}")
            return None

    def prepare_for_session(self, session: 'ProcessingSession') -> None:
        """Prepare keyboard injector for processing a specific session."""
        self._session_window_id = session.recording_session.window_id
        if self.debug_enabled and self._session_window_id is not None:
            pr_debug(f"Session window ID set to: {self._session_window_id}")

        self._session_window_activated_event = session.window_activated

    def cleanup_session(self) -> None:
        """Clean up session-specific state after processing completes."""
        self._session_window_id = None
        self._session_window_activated_event = None

    def _get_current_window_id(self) -> Optional[str]:
        """Get current active window ID."""
        if self.test_mode:
            return None

        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow"],
                check=True,
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def is_session_window_active(self) -> bool:
        """Check if session window is currently active. Returns True if no session window set."""
        if self._session_window_id is None:
            return True

        current_window = self._get_current_window_id()
        if current_window is None:
            return False

        return current_window == self._session_window_id

    def activate_window(self, window_id: str) -> None:
        """
        Activate specified window and wait for activation with timeout.

        Args:
            window_id: X11 window ID to activate

        Raises:
            RuntimeError: Window activation timeout or failure
        """
        try:
            subprocess.run(
                ["xdotool", "windowactivate", "--sync", window_id],
                check=True,
                capture_output=True,
                text=True
            )

            subprocess.run(
                ["xdotool", "windowfocus", "--sync", window_id],
                check=True,
                capture_output=True,
                text=True
            )

            start_time = time.time()
            max_wait = 2.0
            poll_interval = 0.1

            current_active = None
            while time.time() - start_time < max_wait and current_active != window_id:
                result = subprocess.run(
                    ["xdotool", "getactivewindow"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                current_active = result.stdout.strip()
                if current_active != window_id:
                    time.sleep(poll_interval)

            if current_active != window_id:
                raise RuntimeError(f"Failed to activate window {window_id} after {max_wait} seconds")

            time.sleep(self.WINDOW_ACTIVATION_STABILIZATION_DELAY)

            # Signal session that window is activated and stable for keyboard output
            if self._session_window_activated_event is not None:
                self._session_window_activated_event.set()
                if self.debug_enabled:
                    pr_debug(f"Window activation event signaled for window {window_id}")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Window activation failed for window {window_id}: {e}")

    def _wait_for_session_window(self) -> None:
        """Block until active window matches session window with timeout."""
        if self._session_window_id is None:
            return

        waiting_logged = False
        current_window = self._get_current_window_id()
        start_time = time.time()
        max_wait = 0.5

        while (current_window != self._session_window_id or current_window is None) and (time.time() - start_time < max_wait):
            if current_window is None:
                if self.debug_enabled and not waiting_logged:
                    pr_debug("Cannot determine current window, waiting...")
                    waiting_logged = True
            else:
                if self.debug_enabled and not waiting_logged:
                    pr_debug(f"Waiting for window {self._session_window_id} (current: {current_window})")
                    waiting_logged = True

            time.sleep(0.1)
            current_window = self._get_current_window_id()

        if current_window != self._session_window_id or current_window is None:
            raise RuntimeError(f"Window activation timeout: waited {max_wait} seconds for window {self._session_window_id}, current window is {current_window}")

        if waiting_logged and self.debug_enabled:
            pr_debug("Window restored, proceeding")

        if waiting_logged:
            time.sleep(self.WINDOW_ACTIVATION_STABILIZATION_DELAY)

    def _run_xdotool(self, cmd: list) -> None:
        """Execute xdotool command after waiting for modifier keys to be released."""
        self._modifier_tracker.wait_for_no_modifiers()
        self._wait_for_session_window()

        try:
            if self.debug_enabled:
                pr_debug(f"xdotool command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            pr_err(f"xdotool command failed: {str(e)}")

    def bksp(self, count: int) -> None:
        """Backspace count characters."""
        if self.test_mode or count <= 0:
            return

        cmd = [
            "xdotool", "key",
            "--delay", str(self.typing_delay),
            "--repeat", str(count),
            "BackSpace"
        ]
        self._run_xdotool(cmd)
    
    def emit(self, text: str) -> None:
        """Emit text at current cursor position."""
        if self.test_mode or not text:
            return

        lines = text.split('\n')
        for i, line in enumerate(lines):
            if line:
                cmd = [
                    "xdotool", "type",
                    "--delay", str(self.typing_delay),
                    "--",
                    line
                ]
                self._run_xdotool(cmd)

            if i < len(lines) - 1:
                cmd = ["xdotool", "key", "Return"]
                self._run_xdotool(cmd)

    def __del__(self):
        """Cleanup modifier tracker on destruction."""
        if hasattr(self, '_modifier_tracker'):
            self._modifier_tracker.stop()