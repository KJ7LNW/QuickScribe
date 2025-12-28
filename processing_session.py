"""
Processing infrastructure for a recording session.
"""
import queue
import threading
from typing import Optional
from recording_session import RecordingSession
from providers.conversation_context import ConversationContext
from audio_source import AudioResult


class ProcessingSession:
    """Processing infrastructure for a recording session."""

    def __init__(
        self,
        recording_session: RecordingSession,
        context: ConversationContext,
        audio_results: list[AudioResult]
    ):
        self.recording_session: RecordingSession = recording_session
        self.context: ConversationContext = context
        self.audio_results: list[AudioResult] = audio_results
        self.chunk_queue: queue.Queue = queue.Queue()
        self.chunks_complete: threading.Event = threading.Event()
        self.window_activated: threading.Event = threading.Event()
        self.error_message: Optional[str] = None
        self.retry_count: int = 0

    @property
    def has_error(self) -> bool:
        """Check if session encountered an error."""
        return self.error_message is not None
