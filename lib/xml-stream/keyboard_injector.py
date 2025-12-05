"""Keyboard injector interface for XML stream processor."""

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from processing_session import ProcessingSession


class KeyboardInjector(ABC):
    """Abstract interface for keyboard injection operations."""

    @abstractmethod
    def bksp(self, count: int) -> None:
        """Backspace count characters."""
        pass

    @abstractmethod
    def emit(self, text: str) -> None:
        """Emit text at current cursor position."""
        pass

    def get_trigger_window_id(self) -> Optional[str]:
        """Get active window ID at trigger time. Platform-specific implementations override."""
        return None

    def is_session_window_active(self) -> bool:
        """Check if session window is currently active. Default True (no window validation)."""
        return True

    def activate_window(self, window_id: str) -> None:
        """
        Activate specified window. Platform-specific implementations override.

        Args:
            window_id: Platform-specific window identifier

        Raises:
            RuntimeError: Window activation failed
        """
        raise NotImplementedError("Window activation not supported on this platform")

    def prepare_for_session(self, session: 'ProcessingSession') -> None:
        """Prepare keyboard injector for processing a specific session."""
        pass

    def cleanup_session(self) -> None:
        """Clean up session-specific state after processing completes."""
        pass


class MockKeyboardInjector(KeyboardInjector):
    """Mock keyboard injector for testing."""
    
    def __init__(self):
        self.output = ""
        self.operations = []
    
    def bksp(self, count: int) -> None:
        """Backspace by removing characters from end of output."""
        self.operations.append(('bksp', count))
        if count > 0:
            self.output = self.output[:-count]
    
    def emit(self, text: str) -> None:
        """Emit text by appending to output."""
        self.operations.append(('emit', text))
        self.output += text
    
    def reset(self) -> None:
        """Reset mock state."""
        self.output = ""
        self.operations = []