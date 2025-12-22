"""Base class for transcription audio sources."""

import sys
import time
import numpy as np
from abc import abstractmethod
from typing import Optional
from audio_source import AudioResult, AudioTextResult, AudioChunkHandler
from microphone_audio_source import MicrophoneAudioSource

sys.path.insert(0, 'lib')
from pr_log import pr_info


class TranscriptionAudioSource(MicrophoneAudioSource):
    """
    Base class for audio sources that transcribe audio to text.

    Supports both streaming (via AudioChunkHandler) and batch processing.
    Subclasses implement _transcribe_audio() to perform actual transcription.
    """

    def __init__(self, config, model_identifier: str, supports_streaming: bool = False,
                 dtype: str = 'int16', chunk_handler: Optional[AudioChunkHandler] = None):
        """
        Initialize transcription audio source.

        Args:
            config: Configuration object
            model_identifier: Model path or identifier
            supports_streaming: Whether this implementation supports streaming
            dtype: Audio data type
            chunk_handler: Optional chunk handler for streaming implementations
        """
        super().__init__(config, dtype, chunk_handler)
        self.model_identifier = model_identifier
        self.supports_streaming = supports_streaming

    @abstractmethod
    def _transcribe_audio(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio data to text.

        Subclasses implement the actual transcription logic.
        Timing is handled by the base class.

        Args:
            audio_data: Audio data array

        Returns:
            Transcribed text (without formatting)
        """
        pass

    def format_output(self, text: str) -> str:
        """
        Format transcription output with TX tags.

        Template method for wrapping transcription result.
        Subclasses may override for custom formatting (e.g., speed attributes).

        Args:
            text: Transcribed text

        Returns:
            Formatted text with TX tags
        """
        result = ""

        if text:
            result = f"<tx>{text}</tx>"

        return result

    def stop_recording(self) -> AudioResult:
        """
        Stop recording and return audio data immediately.

        Transcription is deferred to worker thread to avoid blocking recording queue.
        Use transcribe_audio_data() in worker thread to perform transcription.
        """
        audio_result = super().stop_recording()

        if self.chunk_handler and hasattr(self.chunk_handler, 'end_streaming'):
            if hasattr(audio_result, 'audio_data') and len(audio_result.audio_data) > 0:
                self.chunk_handler.end_streaming()

        return audio_result

    def transcribe_audio_data(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio data to formatted text.

        Called from worker thread to perform transcription asynchronously.
        Handles timing measurements and output formatting.

        Args:
            audio_data: Audio data array

        Returns:
            Formatted transcription text with TX tags
        """
        if len(audio_data) == 0:
            return ""

        pr_info(f"Transcribing with {self.model_identifier}...")

        transcription_start_time = time.time()
        transcribed_text = self._transcribe_audio(audio_data)
        transcription_end_time = time.time()

        elapsed_ms = int((transcription_end_time - transcription_start_time) * 1000)
        pr_info(f"Transcription completed ({elapsed_ms}ms)")

        formatted_text = self.format_output(transcribed_text)
        return formatted_text

    @staticmethod
    def normalize_to_float32(audio_data: np.ndarray) -> np.ndarray:
        """Convert audio data to float32 normalized to [-1, 1]."""
        if audio_data.dtype != np.float32:
            return audio_data.astype(np.float32) / 32768.0
        return audio_data

    @staticmethod
    def validate_audio_length(audio_data: np.ndarray, sample_rate: int, min_duration_ms: int = 20) -> bool:
        """Check if audio meets minimum length requirement."""
        min_samples = max(320, sample_rate * min_duration_ms // 1000)
        return len(audio_data) >= min_samples

    @staticmethod
    def squeeze_to_mono(audio_data: np.ndarray) -> np.ndarray:
        """Ensure audio is 1D by squeezing out channel dimension."""
        if audio_data.ndim > 1:
            return np.squeeze(audio_data)
        return audio_data
