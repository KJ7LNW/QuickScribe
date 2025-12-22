"""
Shared LiteLLM utilities for response processing.

Single point of truth for:
- Audio encoding
- Streaming response processing
- Content extraction from chunks
"""
import base64
import io
from typing import Optional

try:
    import soundfile as sf
except ImportError:
    sf = None

from lib.pr_log import pr_notice, pr_err, get_streaming_handler


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


def extract_text(chunk) -> Optional[str]:
    """Extract text content from LiteLLM response chunk."""
    delta = chunk.choices[0].delta
    if delta.content is not None:
        return delta.content
    return None


def extract_reasoning(chunk) -> Optional[str]:
    """Extract reasoning content from LiteLLM response chunk."""
    delta = chunk.choices[0].delta
    if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
        return delta.reasoning_content
    return None


def extract_thinking(chunk) -> Optional[list]:
    """Extract thinking blocks from LiteLLM response chunk."""
    delta = chunk.choices[0].delta
    if hasattr(delta, 'thinking_blocks') and delta.thinking_blocks is not None:
        return delta.thinking_blocks
    return None


def extract_usage(chunk) -> Optional[dict]:
    """Extract usage statistics from LiteLLM response chunk."""
    if hasattr(chunk, 'usage') and chunk.usage is not None:
        return chunk.usage
    return None


def process_streaming_response(response) -> str:
    """
    Process LiteLLM streaming response with output display.

    Displays reasoning, thinking, and output sections.
    Returns accumulated text content.

    Args:
        response: LiteLLM streaming response iterator

    Returns:
        Accumulated text content
    """
    accumulated_text = ""
    reasoning_header_shown = False
    thinking_header_shown = False
    output_header_shown = False

    try:
        with get_streaming_handler() as stream:
            for chunk in response:

                reasoning = extract_reasoning(chunk)
                if reasoning is not None:
                    if not reasoning_header_shown:
                        pr_notice("[REASONING]")
                        reasoning_header_shown = True
                    stream.write(reasoning)

                thinking = extract_thinking(chunk)
                if thinking is not None:
                    if not thinking_header_shown:
                        pr_notice("[THINKING]")
                        thinking_header_shown = True
                    for block in thinking:
                        if 'thinking' in block:
                            stream.write(block['thinking'])

                text = extract_text(chunk)
                if text is not None:
                    if not output_header_shown:
                        pr_notice("[OUTPUT]")
                        output_header_shown = True
                    stream.write(text)
                    accumulated_text += text

    except Exception as e:
        pr_err(f"Error processing streaming response: {e}")
        raise

    return accumulated_text
