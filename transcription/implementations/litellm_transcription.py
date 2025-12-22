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
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'lib'))
from pr_log import pr_err, pr_info, pr_debug

from transcription.base import TranscriptionAudioSource
from providers.mapper_factory import MapperFactory
from providers.registry import extract_provider
from instruction_composer import InstructionComposer
from providers.litellm_utils import encode_audio_to_base64, process_streaming_response


class LiteLLMTranscriptionAudioSource(TranscriptionAudioSource):
    """
    LiteLLM-based transcription for cloud API backends with audio support.

    Uses literal transcription instructions to produce verbatim text output.
    Streams response with reasoning/thinking display for consistency with main provider.
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

        # Load literal transcription instructions via composer (cached)
        composer = InstructionComposer()
        self.instructions = composer.load_file('transcription/literal.md')
        if self.instructions is None:
            raise FileNotFoundError("Literal transcription instructions not found: transcription/literal.md")

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

    def _transcribe_audio(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio using LiteLLM provider with streaming output.

        Args:
            audio_data: Audio data array

        Returns:
            Transcribed text (plain text, no formatting)
        """
        # Encode audio to base64 using shared utility
        audio_b64 = encode_audio_to_base64(audio_data, self.config.sample_rate)

        # Build audio content using provider-specific mapper
        audio_content = self.mapper.map_audio_params(audio_b64, "wav")

        # Build messages
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": [
                {"type": "text", "text": "Transcribe the audio following system instructions."},
                audio_content
            ]}
        ]

        # Build completion params
        completion_params = {
            "model": self.model_without_route,
            "messages": messages,
            "stream": True,
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

        # Call LiteLLM with streaming
        try:
            pr_info("RECEIVED FROM TRANSCRIPTION MODEL (streaming):")
            response = self.litellm.completion(**completion_params)

            # Process streaming response using shared utility
            accumulated_text = process_streaming_response(response)

            if accumulated_text is None:
                accumulated_text = ""
            else:
                accumulated_text = accumulated_text.strip()

            return accumulated_text

        except Exception as e:
            pr_err(f"LiteLLM transcription error: {e}")
            raise
