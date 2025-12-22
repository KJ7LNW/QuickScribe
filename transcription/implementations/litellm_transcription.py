"""
LiteLLM-based transcription provider for cloud API backends.

Supports any LiteLLM provider with audio input capability:
- Gemini (gemini-2.5-flash, etc.)
- OpenAI (gpt-4o-audio-preview, etc.)
- Anthropic (claude-3-5-sonnet, etc.)
- Groq
- OpenRouter
"""
import sys
import base64
import io
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'lib'))
from pr_log import pr_err, pr_info, pr_debug

try:
    import soundfile as sf
except ImportError:
    sf = None

from transcription.base import TranscriptionAudioSource
from providers.mapper_factory import MapperFactory
from providers.registry import extract_provider
from instruction_composer import InstructionComposer


class LiteLLMTranscriptionAudioSource(TranscriptionAudioSource):
    """
    LiteLLM-based transcription for cloud API backends with audio support.

    Uses literal transcription instructions to produce verbatim text output.
    """

    def __init__(self, config):
        """
        Initialize LiteLLM transcription source.

        Args:
            config: Configuration object with transcription_model attribute
        """
        model_identifier = config.transcription_model
        super().__init__(
            config,
            model_identifier=model_identifier,
            supports_streaming=False,
            dtype='int16'
        )

        self.provider = extract_provider(model_identifier)

        if not self.provider:
            raise ValueError(
                f"Invalid transcription model format: '{model_identifier}'. "
                "Expected format: provider/model"
            )

        # Extract model without route
        if '@' in model_identifier:
            self.model_without_route = model_identifier.split('@', 1)[0]
            self.route = model_identifier.split('@', 1)[1]
        else:
            self.model_without_route = model_identifier
            self.route = None

        # Get provider-specific mapper
        self.mapper = MapperFactory.get_mapper(self.provider)

        # Instruction composer for cached loading with auto-reload
        self.instruction_composer = InstructionComposer()

        # Import LiteLLM
        try:
            import litellm
            self.litellm = litellm
        except ImportError:
            raise ImportError("litellm library required for LiteLLM transcription. Install with: pip install litellm")

        if config.litellm_debug:
            pr_debug("Enabling LiteLLM debug logging for transcription")
            self.litellm._turn_on_debug()

        pr_info(f"LiteLLM transcription initialized: {model_identifier}")

    def _encode_audio_to_base64(self, audio_np: np.ndarray, sample_rate: int) -> str:
        """
        Encode audio numpy array to base64 WAV string.

        Args:
            audio_np: Audio data as numpy array
            sample_rate: Sample rate in Hz

        Returns:
            Base64-encoded WAV data
        """
        if sf is None:
            raise ImportError("soundfile library required for audio encoding. Install with: pip install soundfile")

        wav_bytes_io = io.BytesIO()
        sf.write(wav_bytes_io, audio_np, sample_rate, format='WAV', subtype='PCM_16')
        wav_bytes = wav_bytes_io.getvalue()
        wav_bytes_io.close()
        return base64.b64encode(wav_bytes).decode('utf-8')

    def _transcribe_audio(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio using LiteLLM provider.

        Args:
            audio_data: Audio data array

        Returns:
            Transcribed text (plain text, no formatting)
        """
        # Load instructions with caching and auto-reload on modification
        instructions = self.instruction_composer._load('transcription/literal.md')
        if instructions is None:
            raise FileNotFoundError("Literal transcription instructions not found: instructions/transcription/literal.md")

        # Encode audio to base64
        audio_b64 = self._encode_audio_to_base64(audio_data, self.config.sample_rate)

        # Build audio content using provider-specific mapper
        audio_content = self.mapper.map_audio_params(audio_b64, "wav")

        # Build messages
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": [audio_content]}
        ]

        # Build completion params
        completion_params = {
            "model": self.model_without_route,
            "messages": messages,
            "stream": False,
            "timeout": self.config.http_timeout
        }

        # Add route parameters if present
        if self.route:
            route_params = self.mapper.map_route_to_completion_params(self.route)
            completion_params.update(route_params)

        # Add max_tokens if configured
        if self.config.max_tokens is not None:
            completion_params["max_tokens"] = self.config.max_tokens

        # Add API key if provided
        if self.config.api_key:
            completion_params["api_key"] = self.config.api_key

        # Map reasoning parameters via provider-specific mapper
        if self.mapper.supports_reasoning(self.model_without_route):
            reasoning_params = self.mapper.map_reasoning_params(
                self.config.enable_reasoning,
                self.config.thinking_budget
            )
            completion_params.update(reasoning_params)

        # Call LiteLLM
        try:
            response = self.litellm.completion(**completion_params)
            text = response.choices[0].message.content

            if text is None:
                text = ""
            else:
                text = text.strip()

            return text

        except Exception as e:
            pr_err(f"LiteLLM transcription error: {e}")
            raise
