"""
Shared LiteLLM utilities for response processing.

Single point of truth for:
- Audio encoding
- Streaming response processing
- Content extraction from chunks
"""
import base64
import io
from typing import Protocol, Optional, Any, Callable

try:
    import soundfile as sf
except ImportError:
    sf = None

from lib.pr_log import pr_notice, pr_err, get_streaming_handler


class StreamExtractor(Protocol):
    """Protocol for provider-specific response chunk extraction."""

    def extract_text(self, chunk) -> Optional[str]:
        """Extract text content from response chunk."""
        ...

    def extract_reasoning(self, chunk) -> Optional[str]:
        """Extract reasoning content from response chunk."""
        ...

    def extract_thinking(self, chunk) -> Optional[list]:
        """Extract thinking blocks from response chunk."""
        ...

    def extract_usage(self, chunk) -> Optional[dict]:
        """Extract usage statistics from response chunk."""
        ...


def encode_audio_to_base64(audio_np, sample_rate: int) -> str:
    """
    Encode audio numpy array to base64 WAV string.

    Args:
        audio_np: Audio data as numpy array
        sample_rate: Sample rate in Hz

    Returns:
        Base64-encoded WAV data

    Raises:
        ImportError: If soundfile library not available
    """
    if sf is None:
        raise ImportError("soundfile library required for audio encoding. Install with: pip install soundfile")

    wav_bytes_io = io.BytesIO()
    sf.write(wav_bytes_io, audio_np, sample_rate, format='WAV', subtype='PCM_16')
    wav_bytes = wav_bytes_io.getvalue()
    wav_bytes_io.close()
    return base64.b64encode(wav_bytes).decode('utf-8')


def litellm_extract_text(chunk) -> Optional[str]:
    """Extract text content from LiteLLM response chunk."""
    delta = chunk.choices[0].delta
    if delta.content is not None:
        return delta.content
    return None


def litellm_extract_reasoning(chunk) -> Optional[str]:
    """Extract reasoning content from LiteLLM response chunk."""
    delta = chunk.choices[0].delta
    if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
        return delta.reasoning_content
    return None


def litellm_extract_thinking(chunk) -> Optional[list]:
    """Extract thinking blocks from LiteLLM response chunk."""
    delta = chunk.choices[0].delta
    if hasattr(delta, 'thinking_blocks') and delta.thinking_blocks is not None:
        return delta.thinking_blocks
    return None


def litellm_extract_usage(chunk) -> Optional[dict]:
    """Extract usage statistics from LiteLLM response chunk."""
    if hasattr(chunk, 'usage') and chunk.usage is not None:
        return chunk.usage
    return None


class LiteLLMExtractor:
    """Extractor implementation using module-level LiteLLM functions."""

    def extract_text(self, chunk) -> Optional[str]:
        return litellm_extract_text(chunk)

    def extract_reasoning(self, chunk) -> Optional[str]:
        return litellm_extract_reasoning(chunk)

    def extract_thinking(self, chunk) -> Optional[list]:
        return litellm_extract_thinking(chunk)

    def extract_usage(self, chunk) -> Optional[dict]:
        return litellm_extract_usage(chunk)


def stream_response(
    response,
    extractor: StreamExtractor,
    on_chunk: Optional[Callable[[str, Any, str], None]] = None
) -> tuple[str, Optional[dict], object]:
    """
    Process streaming response with provider-specific extraction.

    Core streaming logic: iteration, extraction, display, accumulation.
    Single point of truth for streaming behavior.

    Args:
        response: Streaming response iterator
        extractor: Provider implementing StreamExtractor protocol
        on_chunk: Optional callback(chunk_type, content, accumulated_text) for orchestration.
                  Return False to stop iteration, None/True to continue.

    Returns:
        Tuple of (accumulated_text, usage_data, last_chunk)
    """
    accumulated_text = ""
    usage_data = None
    last_chunk = None
    reasoning_header_shown = False
    thinking_header_shown = False
    output_header_shown = False

    with get_streaming_handler() as stream:
        for chunk in response:
            last_chunk = chunk

            # Extract and display reasoning
            reasoning = extractor.extract_reasoning(chunk)
            if reasoning is not None:
                if not reasoning_header_shown:
                    pr_notice("[REASONING]")
                    reasoning_header_shown = True
                stream.write(reasoning)
                if on_chunk:
                    result = on_chunk('reasoning', reasoning, accumulated_text)
                    if result is False:
                        break

            # Extract and display thinking
            thinking = extractor.extract_thinking(chunk)
            if thinking is not None:
                if not thinking_header_shown:
                    pr_notice("[THINKING]")
                    thinking_header_shown = True
                for block in thinking:
                    if 'thinking' in block:
                        stream.write(block['thinking'])
                if on_chunk:
                    result = on_chunk('thinking', thinking, accumulated_text)
                    if result is False:
                        break

            # Extract and display text
            text = extractor.extract_text(chunk)
            if text is not None:
                if not output_header_shown:
                    pr_notice("[OUTPUT]")
                    output_header_shown = True
                stream.write(text)
                accumulated_text += text
                if on_chunk:
                    result = on_chunk('text', text, accumulated_text)
                    if result is False:
                        break

            # Extract usage statistics
            usage = extractor.extract_usage(chunk)
            if usage is not None:
                usage_data = usage
                if on_chunk:
                    result = on_chunk('usage', usage, accumulated_text)
                    if result is False:
                        break

    return (accumulated_text, usage_data, last_chunk)


def process_streaming_response(response) -> str:
    """
    Process LiteLLM streaming response with output display.

    Wrapper for stream_response() using module-level extractors.
    Displays reasoning, thinking, and output sections.
    Returns accumulated text content.

    Args:
        response: LiteLLM streaming response iterator

    Returns:
        Accumulated text content
    """
    try:
        accumulated_text, _, _ = stream_response(response, LiteLLMExtractor())
        return accumulated_text
    except Exception as e:
        pr_err(f"Error processing streaming response: {e}")
        raise
