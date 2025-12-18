"""NeMo TDT transcription implementation for QuickScribe."""

import sys
import numpy as np
import tempfile
import os

sys.path.insert(0, 'lib')
from pr_log import pr_err, pr_warn, pr_info

try:
    import nemo.collections.asr as nemo_asr
except ImportError:
    nemo_asr = None

from transcription.base import TranscriptionAudioSource


class NeMoTDTTranscriptionAudioSource(TranscriptionAudioSource):
    """
    NeMo TDT (Token-and-Duration Transducer) transcription implementation.

    Supports NVIDIA NeMo ASR models like Parakeet-TDT that use transducer
    architecture. NeMo models handle audio preprocessing internally, so
    processor parameter is unused.
    """

    def __init__(self, config, model, processor):
        """
        Initialize NeMo TDT transcription audio source.

        Args:
            config: Configuration object
            model: Pre-loaded nemo.collections.asr.models.ASRModel instance
            processor: Unused (None) - NeMo handles preprocessing internally
        """
        if nemo_asr is None:
            raise ImportError("NeMo toolkit not installed")

        try:
            model_identifier = model.cfg.name
        except AttributeError:
            model_identifier = "nemo-tdt-model"

        super().__init__(config, model_identifier, supports_streaming=False, dtype='float32')

        self.model = model

        pr_info(f"Initialized NeMo TDT model: {model_identifier}")

    def _transcribe_audio(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio using NeMo TDT model.

        NeMo requires audio as file path. Writes numpy array to temporary
        WAV file in workspace directory, transcribes, then removes file.

        Args:
            audio_data: Audio data array

        Returns:
            Transcribed text
        """
        try:
            audio_data = self.normalize_to_float32(audio_data)
            audio_data = self.squeeze_to_mono(audio_data)

            if not self.validate_audio_length(audio_data, self.config.sample_rate):
                pr_warn("Audio too short for NeMo transcription")
                return ""

            temp_wav_path = self._write_temp_wav(audio_data)

            try:
                output = self.model.transcribe([temp_wav_path])

                if output and len(output) > 0:
                    transcription = output[0].text
                else:
                    transcription = ""

                return transcription.strip()

            finally:
                if os.path.exists(temp_wav_path):
                    os.remove(temp_wav_path)

        except Exception as e:
            pr_err(f"Error during NeMo transcription: {e}")
            return ""

    def _write_temp_wav(self, audio_data: np.ndarray) -> str:
        """
        Write audio data to temporary WAV file in workspace.

        Args:
            audio_data: Float32 audio data normalized to [-1, 1]

        Returns:
            Path to temporary WAV file
        """
        import wave

        fd, temp_path = tempfile.mkstemp(suffix='.wav', dir='.', prefix='nemo_audio_')

        try:
            os.close(fd)

            audio_int16 = (audio_data * 32767).astype(np.int16)

            with wave.open(temp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.config.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())

            return temp_path

        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    def initialize(self) -> bool:
        """Initialize NeMo transcription source."""
        try:
            if nemo_asr is None:
                pr_err("NeMo toolkit not available")
                pr_err("Install with: pip install nemo_toolkit[asr]")
                return False

            if not super().initialize():
                return False

            pr_info(f"NeMo TDT model initialized: {self.model_identifier}")
            return True

        except Exception as e:
            pr_err(f"Error initializing NeMo model: {e}")
            return False
