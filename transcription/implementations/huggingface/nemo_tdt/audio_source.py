"""NeMo TDT transcription implementation for QuickScribe."""

import math
import sys
import numpy as np
import tempfile
import os
from copy import deepcopy

from omegaconf import open_dict

sys.path.insert(0, 'lib')
from pr_log import pr_err, pr_warn, pr_info

try:
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.utils.streaming_utils import BatchedFrameASRTDT
except ImportError:
    nemo_asr = None
    BatchedFrameASRTDT = None

from transcription.base import TranscriptionAudioSource

# Duration threshold in seconds above which buffered inference is used
BUFFERED_INFERENCE_THRESHOLD_SECS = 60.0


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

        # Streaming parameters derived from model config
        window_stride = model.cfg.preprocessor.window_stride
        subsampling_factor = model.cfg.get("subsampling_factor", 8)
        self.model_stride_in_secs = window_stride * subsampling_factor
        self.frame_len = 1.6
        self.total_buffer = 4.0
        self.tokens_per_chunk = math.ceil(self.frame_len / self.model_stride_in_secs)
        self.mid_delay = math.ceil(
            (self.frame_len + (self.total_buffer - self.frame_len) / 2) / self.model_stride_in_secs
        )

        pr_info(f"Initialized NeMo TDT model: {model_identifier}")

    def _transcribe_audio(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio using NeMo TDT model.

        For short audio, writes to temporary WAV and calls model.transcribe()
        directly. For audio exceeding BUFFERED_INFERENCE_THRESHOLD_SECS,
        uses FrameBatchASR to process in fixed-size frames with bounded
        GPU memory.

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

            duration_secs = len(audio_data) / self.config.sample_rate

            if duration_secs > BUFFERED_INFERENCE_THRESHOLD_SECS:
                pr_info(f"Using buffered inference for {duration_secs:.1f}s audio")
                return self._transcribe_audio_buffered(audio_data)

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

    def _transcribe_audio_buffered(self, audio_data: np.ndarray) -> str:
        """
        Transcribe long audio using NeMo BatchedFrameASRTDT chunked inference.

        Preserves acoustic context via total_buffer overlap and language model
        context via stateful LSTM decoding across chunk boundaries. GPU memory
        is bounded to one chunk's encoder activations at a time.

        Temporarily switches the model's decoding strategy to greedy with
        preserved alignments (required by BatchedFrameASRTDT), then restores
        the original config afterward. This is intentional config switching
        for a shared model resource, not a callee-mutation repair pattern.

        Args:
            audio_data: Float32 mono audio data normalized to [-1, 1]

        Returns:
            Transcribed text
        """
        temp_wav_path = self._write_temp_wav(audio_data)

        # Save original decoding config before switching to streaming mode
        saved_decoding_cfg = deepcopy(self.model.cfg.decoding)

        try:
            with open_dict(self.model.cfg.decoding):
                self.model.cfg.decoding.strategy = "greedy"
                self.model.cfg.decoding.preserve_alignments = True
                self.model.cfg.decoding.fused_batch_size = -1
            self.model.change_decoding_strategy(self.model.cfg.decoding)

            frame_asr = BatchedFrameASRTDT(
                asr_model=self.model,
                frame_len=self.frame_len,
                total_buffer=self.total_buffer,
                batch_size=1,
            )

            # NeMo 2.6.0 bug: BatchedFrameASRTDT.__init__ does not forward
            # stateful_decoding to parent BatchedFrameASRRNNT.__init__
            frame_asr.stateful_decoding = True

            frame_asr.read_audio_file(
                [temp_wav_path],
                delay=self.mid_delay,
                model_stride_in_secs=self.model_stride_in_secs,
            )

            hypotheses = frame_asr.transcribe(
                tokens_per_chunk=self.tokens_per_chunk,
                delay=self.mid_delay,
            )

            if len(hypotheses) == 0:
                return ""

            return hypotheses[0].strip()

        finally:
            # Restore original decoding config so short-path model.transcribe() is unaffected
            with open_dict(self.model.cfg.decoding):
                self.model.cfg.decoding = saved_decoding_cfg
            self.model.change_decoding_strategy(self.model.cfg.decoding)

            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)

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
