"""
Abstract base provider class with common infrastructure.

Provides template method pattern for provider implementations:
- Shared: streaming, timing, instructions, text building, error handling
- Hooks: _generate_response(), _extract_text(), _extract_reasoning(), _extract_usage()
- Concrete providers: LiteLLMProvider, HuggingFaceProvider
"""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import time
import sys
from .conversation_context import ConversationContext
from instruction_composer import InstructionComposer
from lib.pr_log import (
    pr_emerg, pr_alert, pr_crit, pr_err, pr_warn, pr_notice, pr_info, pr_debug,
    get_streaming_handler
)


class TerminateStream(Exception):
    """Signal to terminate streaming when </xml> tag is detected."""
    pass


class StreamTimeout(Exception):
    """Signal timeout waiting for streaming response chunks."""
    pass


class AbstractProvider(ABC):
    """
    Abstract base provider for LLM backends.

    Implements template method pattern with shared infrastructure:
    - Streaming protocol and callback handling
    - Timing measurement
    - Instruction composition
    - Text building for prompts
    - Error handling

    Subclasses implement provider-specific hooks:
    - initialize(): Setup and validation
    - _generate_response(): Generate response stream
    - _extract_text(): Extract text from response chunks
    - _extract_reasoning(): Extract reasoning content (optional)
    - _extract_usage(): Extract usage statistics (optional)
    """

    def __init__(self, config, audio_processor):
        if audio_processor is None:
            raise ValueError("audio_processor is required and cannot be None")

        self.config = config
        self._initialized = False

        # Timing tracking
        self.model_start_time = None
        self.first_response_time = None

        # Audio processor for instruction injection
        self.audio_processor = audio_processor

        # Instruction composition
        self.instruction_composer = InstructionComposer()

        # Route extraction (format: provider/model@route)
        if '@' in config.model_id:
            model_parts = config.model_id.split('@', 1)
            self.model_without_route: str = model_parts[0]
            self.route: Optional[str] = model_parts[1]
        else:
            self.model_without_route: str = config.model_id
            self.route: Optional[str] = None

        # Provider extraction (single point of truth)
        self.provider = self._extract_provider(self.model_without_route)

    def _extract_provider(self, model_without_route: str) -> str:
        """Extract provider from model (format: provider/model)."""
        if '/' in model_without_route:
            return model_without_route.split('/', 1)[0].lower()
        return ''

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize provider and validate access.

        Provider-specific implementation.
        Returns True if initialization successful.
        """
        pass

    @abstractmethod
    def _generate_response(self, instructions: str, context, audio_data, text_data):
        """
        Generate response stream from provider.

        Provider-specific implementation.
        Yields chunks in provider-specific format.
        """
        pass

    def _extract_text(self, chunk) -> Optional[str]:
        """Extract text content from response chunk."""
        return None

    def _extract_reasoning(self, chunk) -> Optional[str]:
        """Extract reasoning content from response chunk."""
        return None

    def _extract_usage(self, chunk) -> Optional[dict]:
        """Extract usage statistics from response chunk."""
        return None

    def _extract_thinking(self, chunk) -> Optional[list]:
        """Extract thinking blocks from response chunk."""
        return None

    def _build_context_text(self, context) -> str:
        """
        Build context portion of prompt (shared by all providers).

        Args:
            context: ConversationContext with xml_markup and compiled_text

        Returns:
            Context text string
        """
        if context.xml_markup:
            return (
                f"Current conversation XML: {context.xml_markup}\n"
                f"Current conversation text: {context.compiled_text}"
            )

        return (
            "CRITICAL: No prior conversation. Treat input as audio dictation and follow system instructions."
        )

    def _build_text_input_explanation(self, text_data: str) -> str:
        """
        Build text input explanation (shared when using transcription mode).

        Args:
            text_data: Transcribed text from audio

        Returns:
            Input explanation string with transcription guidance
        """
        return (
            f"NEW INPUT (requires processing):\n"
            f"Mechanical transcription: {text_data}\n\n"
            "CRITICAL: The 'mechanical transcription' above is raw output "
            "from automatic speech recognition. Treat as audio input and follow system instructions."
        )

    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._initialized

    def _process_streaming_response(self, response, streaming_callback=None, final_callback=None):
        """
        Process streaming response chunks using extraction hooks.

        Template method that handles streaming protocol:
        - Extracts content via provider hooks
        - Manages callbacks
        - Detects </xml> termination
        - Tracks timing
        """
        pr_info("RECEIVED FROM MODEL (streaming):")
        accumulated_text = ""
        usage_data = None
        last_chunk = None
        reasoning_header_shown = False
        thinking_header_shown = False
        output_header_shown = False

        try:
            with get_streaming_handler() as stream:
                for chunk in response:
                    last_chunk = chunk

                    reasoning = self._extract_reasoning(chunk)
                    if reasoning is not None:
                        if not reasoning_header_shown:
                            pr_notice("[REASONING]")
                            reasoning_header_shown = True
                        stream.write(reasoning)
                        if streaming_callback:
                            streaming_callback(('keepalive', None))

                    thinking = self._extract_thinking(chunk)
                    if thinking is not None:
                        if not thinking_header_shown:
                            pr_notice("[THINKING]")
                            thinking_header_shown = True
                        for block in thinking:
                            if 'thinking' in block:
                                stream.write(block['thinking'])
                        if streaming_callback:
                            streaming_callback(('keepalive', None))

                    text = self._extract_text(chunk)
                    if text is not None:
                        if not output_header_shown:
                            pr_notice("[OUTPUT]")
                            output_header_shown = True
                        self.mark_first_response()
                        stream.write(text)
                        if streaming_callback:
                            streaming_callback(text)
                        accumulated_text += text

                        if '</xml>' in accumulated_text:
                            raise TerminateStream()

                    usage = self._extract_usage(chunk)
                    if usage is not None:
                        usage_data = usage

        except TerminateStream:
            self._close_response_stream(response)
            pr_debug("Stream terminated: </xml> tag detected")

        self._print_timing_stats()
        self._on_response_complete(usage_data, last_chunk)

        if final_callback:
            final_callback(accumulated_text)

    def _close_response_stream(self, response) -> None:
        """Close response stream if supported."""
        if hasattr(response, 'completion_stream') and hasattr(response.completion_stream, 'close'):
            response.completion_stream.close()

    def _on_response_complete(self, usage_data, last_chunk) -> None:
        """Hook for post-response processing. Override for usage/cost display."""
        pass

    def _display_user_content(self, user_content):
        """Display user content being sent to model (optional, provider-specific)."""
        pass

    def transcribe(self, context: ConversationContext,
                   audio_data: Optional[np.ndarray] = None,
                   text_data: Optional[str] = None,
                   streaming_callback=None,
                   final_callback=None) -> None:
        """
        Unified transcription interface for both audio and text inputs.

        Template method that orchestrates:
        1. Get instructions
        2. Start timing
        3. Call provider-specific _generate_response() hook
        4. Process streaming response with hooks

        Args:
            context: Conversation context with XML markup and compiled text
            audio_data: Optional audio data as numpy array
            text_data: Optional pre-transcribed text
            streaming_callback: Optional callback for streaming text chunks
            final_callback: Optional callback for final result
        """
        if not self.is_initialized():
            pr_err("Provider not initialized.")
            return

        try:
            xml_instructions = self.get_xml_instructions()
            self.start_model_timer()

            response = self._generate_response(xml_instructions, context, audio_data, text_data)
            self._process_streaming_response(response, streaming_callback, final_callback)

        except Exception as e:
            operation = "audio transcription" if audio_data is not None else "text processing"
            self._handle_provider_error(e, operation)

    def get_xml_instructions(self) -> str:
        """Get the composed XML instructions from files."""
        # Determine audio source name for instruction loading
        audio_source_name = None
        if self.config.audio_source in ['transcribe', 'trans']:
            transcription_lower = self.config.transcription_model.lower()
            if 'wav2vec2' in transcription_lower or 'huggingface' in transcription_lower:
                audio_source_name = 'wav2vec2'
            elif 'vosk' in transcription_lower:
                audio_source_name = 'vosk'
            elif 'whisper' in transcription_lower:
                audio_source_name = 'whisper'

        # Compose instructions from files (reads current mode from config)
        instructions = self.instruction_composer.compose(
            self.config.mode,
            audio_source_name,
            self.provider
        )

        return instructions


    def start_model_timer(self):
        """Mark the start of model processing for timing measurements."""
        self.model_start_time = time.time()
        self.first_response_time = None  # Reset for new request

    def mark_first_response(self):
        """Mark when the first response chunk is received."""
        if self.first_response_time is None:
            self.first_response_time = time.time()

    def _print_timing_stats(self):
        """Print timing statistics."""
        if self.model_start_time and self.first_response_time:
            model_time = self.first_response_time - self.model_start_time
            pr_debug(f"Model processing time: {model_time:.3f}s")

    def _handle_provider_error(self, error: Exception, operation: str) -> None:
        """Handle provider errors with logging and re-raise."""
        import traceback

        pr_err(f"ERROR during {operation}:")
        pr_err(f"Error Type: {type(error).__name__}")
        pr_err(f"Error Message: {str(error)}")
        pr_debug("Stack trace:")
        traceback.print_exc(file=sys.stderr)
        raise

    def _get_generation_config(self) -> dict:
        """Get provider-agnostic generation configuration."""
        config = {
            'temperature': self.config.temperature,
            'enable_reasoning': self.config.enable_reasoning,
            'response_format': 'text'
        }

        if self.config.max_tokens is not None:
            config['max_output_tokens'] = self.config.max_tokens

        # top_p not included - using default to avoid conflicting with temperature

        return config
