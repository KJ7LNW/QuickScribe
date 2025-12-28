"""HuggingFace CTC transcription audio source implementation."""

import numpy as np

try:
    import torch
    import transformers
    from transformers import (
        AutoModelForCTC,
        AutoProcessor
    )
    from transformers.utils import is_offline_mode
except ImportError:
    torch = None
    transformers = None
    AutoModelForCTC = None
    AutoProcessor = None
    is_offline_mode = None

from transcription.base import TranscriptionAudioSource
from providers.registry import extract_model
from lib.pr_log import pr_err, pr_warn, pr_info
from ..processor_utils import load_processor_with_fallback


class HuggingFaceCTCTranscriptionAudioSource(TranscriptionAudioSource):
    """HuggingFace CTC transcription implementation."""

    def __init__(self, config, transcription_model_or_model, processor=None):
        """
        Initialize CTC transcription audio source.

        Supports two initialization modes:
        1. Legacy: transcription_model_or_model is string, loads model internally
        2. New: transcription_model_or_model is pre-loaded model, processor required

        Args:
            config: Configuration object
            transcription_model_or_model: Either model path string or pre-loaded model
            processor: Pre-loaded processor (required when passing model object)
        """
        if isinstance(transcription_model_or_model, str):
            model_identifier = extract_model(transcription_model_or_model)
        else:
            model = transcription_model_or_model
            model_identifier = model.name_or_path if hasattr(model, 'name_or_path') else str(model)

        super().__init__(config, model_identifier, supports_streaming=False, dtype='float32')

        if isinstance(transcription_model_or_model, str):
            self._load_model(model_identifier)
        else:
            if processor is None:
                raise ValueError("processor required when passing pre-loaded model")

            self.model = model
            self.processor = processor
            pr_info(f"Using pre-loaded CTC model: {model_identifier}")

    def _load_model(self, model_path: str):
        """Load CTC model and processor."""
        if torch is None or transformers is None:
            raise ImportError("PyTorch and transformers libraries not installed. Install with: pip install torch transformers huggingface_hub")

        try:
            pr_info(f"Loading CTC model: {model_path}")

            offline_mode = is_offline_mode() if is_offline_mode else False

            self.processor = load_processor_with_fallback(
                model_path,
                cache_dir=None,
                force_download=False,
                local_files_only=offline_mode
            )

            self.model = AutoModelForCTC.from_pretrained(
                model_path,
                cache_dir=None,
                force_download=False,
                local_files_only=offline_mode
            )

            self.model.eval()
            pr_info(f"Successfully loaded CTC model: {model_path}")

        except RuntimeError as e:
            if "has no tokenizer" in str(e):
                raise RuntimeError(str(e)) from None
            raise RuntimeError(f"Failed to load CTC model from {model_path}: {e}") from None
        except Exception as e:
            raise RuntimeError(f"Failed to load CTC model from {model_path}: {e}") from None

    def _transcribe_audio(self, audio_data: np.ndarray) -> str:
        """Transcribe audio using CTC phoneme recognition."""
        return self._process_audio(audio_data)

    def _process_audio(self, audio_data: np.ndarray) -> str:
        """Process single audio variant with CTC model."""
        try:
            if len(audio_data) == 0:
                return ""

            audio_data = self.normalize_to_float32(audio_data)
            audio_data = self.squeeze_to_mono(audio_data)

            if not self.validate_audio_length(audio_data, self.config.sample_rate):
                pr_warn(f"Audio too short for CTC model")
                return ""

            with torch.no_grad():
                input_values = self.processor(
                    audio_data,
                    sampling_rate=self.config.sample_rate,
                    return_tensors="pt"
                ).input_values

                logits = self.model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                raw_output = self.processor.batch_decode(predicted_ids)[0]

                pr_info(f"{self.processor.output_format}: {raw_output}")
                return raw_output

        except Exception as e:
            pr_err(f"Error processing audio with CTC model: {e}")
            return ""

    def initialize(self) -> bool:
        """Initialize HuggingFace CTC audio source."""
        try:
            if torch is None or transformers is None:
                pr_err("PyTorch and transformers libraries not available")
                return False

            if not super().initialize():
                return False

            pr_info(f"HuggingFace CTC initialized with model: {self.model_identifier}")
            return True

        except Exception as e:
            pr_err(f"Error initializing HuggingFace audio source: {e}")
            return False
