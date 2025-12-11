"""
Tests for retry logic in model invocation worker.

Verifies:
- Successful first attempt requires no retry
- Timeout triggers retry up to max attempts
- Max retries exhausted raises error
- InternalServerError does not trigger retry
- Error propagation through session
"""
import unittest
import sys
import os
import time
import threading
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

sys.modules['pynput'] = Mock()
sys.modules['pynput.keyboard'] = Mock()

mock_qt = MagicMock()
sys.modules['PyQt6'] = mock_qt
sys.modules['PyQt6.QtWidgets'] = mock_qt.QtWidgets
sys.modules['PyQt6.QtCore'] = mock_qt.QtCore
sys.modules['PyQt6.QtGui'] = mock_qt.QtGui

from model_invocation_worker import _invoke_model
from processing_session import ProcessingSession
from recording_session import RecordingSession, RecordingSource
from providers.conversation_context import ConversationContext
from audio_source import AudioTextResult
from litellm import exceptions as litellm_exceptions


class MockConfig:
    """Mock configuration for testing."""
    def __init__(self):
        self.connection_timeout = 3.0


class TestRetryLogic(unittest.TestCase):
    """Test retry logic in model_invocation_worker."""

    def test_successful_first_attempt(self):
        """Test that successful first attempt requires no retry."""
        mock_provider = Mock()
        mock_provider.config = MockConfig()

        def mock_transcribe(context, audio_data=None, text_data=None,
                          streaming_callback=None, final_callback=None):
            if streaming_callback:
                streaming_callback("test chunk")

        mock_provider.transcribe = mock_transcribe

        recording = RecordingSession(RecordingSource.KEYBOARD, window_id=None)
        context = ConversationContext(xml_markup="", compiled_text="", sample_rate=16000)
        result = AudioTextResult("test", 16000)
        session = ProcessingSession(recording, context, result)

        _invoke_model(mock_provider, session, text_data="test")

        self.assertTrue(session.chunks_complete.is_set())
        self.assertIsNone(session.error_message)
        self.assertFalse(session.chunk_queue.empty())

    def test_timeout_triggers_retry(self):
        """Test that timeout on first attempt triggers retry."""
        mock_provider = Mock()
        mock_provider.config = MockConfig()
        attempt_count = [0]

        def mock_transcribe(context, audio_data=None, text_data=None,
                          streaming_callback=None, final_callback=None):
            attempt_count[0] += 1
            if attempt_count[0] == 1:
                time.sleep(0.2)
            else:
                if streaming_callback:
                    streaming_callback("success on retry")

        mock_provider.transcribe = mock_transcribe

        recording = RecordingSession(RecordingSource.KEYBOARD, window_id=None)
        context = ConversationContext(xml_markup="", compiled_text="", sample_rate=16000)
        result = AudioTextResult("test", 16000)
        session = ProcessingSession(recording, context, result)

        with patch('lib.connection_monitor.ConnectionMonitor') as MockMonitor:
            monitor_instance = Mock()
            monitor_instance.first_chunk_received = threading.Event()
            monitor_instance.timeout_occurred = False
            MockMonitor.return_value = monitor_instance

            def mark_first_chunk_side_effect():
                monitor_instance.first_chunk_received.set()

            monitor_instance.mark_first_chunk = Mock(side_effect=mark_first_chunk_side_effect)
            monitor_instance.stop = Mock()

            if attempt_count[0] == 0:
                monitor_instance.timeout_occurred = True
            else:
                monitor_instance.first_chunk_received.set()

            _invoke_model(mock_provider, session, text_data="test")

        self.assertTrue(session.chunks_complete.is_set())

    def test_max_retries_exhausted(self):
        """Test that max retries exhausted sets error message."""
        mock_provider = Mock()
        mock_provider.config = MockConfig()

        def mock_transcribe(context, audio_data=None, text_data=None,
                          streaming_callback=None, final_callback=None):
            time.sleep(4.0)

        mock_provider.transcribe = mock_transcribe

        recording = RecordingSession(RecordingSource.KEYBOARD, window_id=None)
        context = ConversationContext(xml_markup="", compiled_text="", sample_rate=16000)
        result = AudioTextResult("test", 16000)
        session = ProcessingSession(recording, context, result)

        _invoke_model(mock_provider, session, text_data="test")

        self.assertTrue(session.chunks_complete.is_set())
        self.assertIsNotNone(session.error_message)
        self.assertIn("timeout", session.error_message.lower())

    def test_internal_server_error_no_retry(self):
        """Test that InternalServerError does not trigger retry."""
        mock_provider = Mock()
        mock_provider.config = MockConfig()
        attempt_count = [0]

        def mock_transcribe(context, audio_data=None, text_data=None,
                          streaming_callback=None, final_callback=None):
            attempt_count[0] += 1
            error = litellm_exceptions.InternalServerError(
                message="Server error",
                llm_provider="test",
                model="test-model"
            )
            raise error

        mock_provider.transcribe = mock_transcribe

        recording = RecordingSession(RecordingSource.KEYBOARD, window_id=None)
        context = ConversationContext(xml_markup="", compiled_text="", sample_rate=16000)
        result = AudioTextResult("test", 16000)
        session = ProcessingSession(recording, context, result)

        _invoke_model(mock_provider, session, text_data="test")

        self.assertEqual(attempt_count[0], 1)
        self.assertTrue(session.chunks_complete.is_set())
        self.assertIsNotNone(session.error_message)


if __name__ == '__main__':
    unittest.main()
