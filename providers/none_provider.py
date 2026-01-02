"""
None provider for TX passthrough without LLM processing.

Extracts first <tx> section from transcription output and converts to update format.
Designed for wav2vec2 multi-speed output or any transcription with <tx> tags.
"""
from typing import Optional
import re
from .base_provider import AbstractProvider
from lib.pr_log import pr_debug


class NoneProvider(AbstractProvider):
    """
    Passthrough provider that bypasses LLM processing.

    Extracts first <tx> tag from transcription and generates <update> tags
    for keyboard injection. Designed for direct phoneme or text passthrough.
    """

    def __init__(self, config, audio_processor):
        super().__init__(config, audio_processor)

    def initialize(self) -> bool:
        """No initialization required for passthrough provider."""
        self._initialized = True
        return True

    def _generate_response(self, instructions: str, context, audio_data, text_data):
        """
        Generate XML response with first TX content converted to update tags.

        Extracts first <tx> section and wraps in complete XML structure.

        Args:
            instructions: System instructions (unused)
            context: Conversation context (unused)
            audio_data: Audio data (unused, text_data required)
            text_data: Transcription text with TX tags

        Yields:
            Complete XML response string
        """
        tx_content = self._extract_first_tx(text_data)
        update_tags = self._text_to_update_tags(tx_content)

        # Build complete XML response
        xml_response = (
            f"<xml>\n"
            f"<tx>{tx_content}</tx>\n"
            f"<update>{update_tags}</update>\n"
            f"</xml>"
        )

        pr_debug(f"NoneProvider generated response: {xml_response}")

        # Yield complete response as single chunk
        yield xml_response

    def _extract_first_tx(self, text_data: str) -> str:
        """
        Extract content from first <tx> tag.

        Pattern matches <tx> or <tx speed="80%"> format.
        Returns first match content, or original text if no tags found.

        Args:
            text_data: Text with optional <tx> tags

        Returns:
            Extracted content or original text
        """
        # Match <tx> or <tx speed="N%"> tags, extract content (first occurrence only)
        match = re.search(r'<tx(?:\s+speed="[^"]*")?>(.*?)</tx>', text_data, re.DOTALL)

        if match:
            result = match.group(1)
        else:
            result = text_data

        return result

    def _text_to_update_tags(self, text: str) -> str:
        """
        Convert text to single update tag.

        Wraps entire content verbatim in tag ID 10 for XMLStreamProcessor.

        Args:
            text: Plain text to convert

        Returns:
            XML update tag (e.g., "<10>exact content</10>")
        """
        result = ""

        if text:
            result = f"<10>{text}</10>"

        return result

    def extract_text(self, chunk) -> Optional[str]:
        """Extract text from chunk (already plain text)."""
        return chunk

    def extract_reasoning(self, chunk) -> Optional[str]:
        """No reasoning in passthrough provider."""
        return None

    def extract_thinking(self, chunk) -> Optional[list]:
        """No thinking blocks in passthrough provider."""
        return None

    def extract_usage(self, chunk) -> Optional[dict]:
        """No usage statistics for passthrough provider."""
        return None
