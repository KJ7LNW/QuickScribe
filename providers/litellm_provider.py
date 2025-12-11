"""
LiteLLM provider for cloud API backends.

Handles: OpenAI, Anthropic, Gemini, Groq, OpenRouter via LiteLLM library.
"""
from typing import Optional
import numpy as np
import base64
import io
import soundfile as sf
from .base_provider import AbstractProvider
from .mapper_factory import MapperFactory
from lib.pr_log import (
    pr_emerg, pr_alert, pr_crit, pr_err, pr_warn, pr_notice, pr_info, pr_debug
)


class LiteLLMProvider(AbstractProvider):
    """
    LiteLLM-based provider for cloud API backends.

    Supports all providers available through LiteLLM:
    - OpenAI (GPT-4, GPT-4o, etc.)
    - Anthropic (Claude)
    - Google (Gemini)
    - Groq
    - OpenRouter
    """

    def __init__(self, config, audio_processor):
        super().__init__(config, audio_processor)

        self.litellm = None
        self.litellm_exceptions = None
        self.mapper = MapperFactory.get_mapper(self.provider)
        self.total_cost = 0.0
        self._validation_results = None

    def initialize(self) -> bool:
        """Initialize LiteLLM and validate model."""
        try:
            import litellm
            from litellm import exceptions
            self.litellm = litellm
            self.litellm_exceptions = exceptions

            if self.config.litellm_debug:
                pr_debug("Enabling LiteLLM debug logging")
                litellm._turn_on_debug()

            if self.config.api_key:
                pr_info(f"Using provided API key for {self.provider}")

            pr_info(f"LiteLLM initialized with model: {self.config.model_id}")

            # Skip validation for transcription-only models
            if self.mapper.uses_transcription_endpoint(self.model_without_route):
                pr_info("Skipping validation for transcription-only model")
                self._initialized = True
                return True

            # Skip validation when using local transcription
            if self.config.audio_source in ['transcribe', 'trans']:
                pr_info("Skipping validation when using local transcription")
                self._initialized = True
                return True

            # Generate minimal test audio (0.1 second silence)
            test_audio_silence = np.zeros(int(0.1 * self.config.sample_rate), dtype=np.int16)
            test_audio_silence_b64 = self._encode_audio_to_base64(test_audio_silence, self.config.sample_rate)

            # Load sumtest.wav for audio intelligence test
            import os
            sumtest_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'samples', 'sumtest.wav')
            sumtest_audio, sumtest_sr = sf.read(sumtest_path)
            if sumtest_audio.dtype != np.int16:
                sumtest_audio = (sumtest_audio * 32767).astype(np.int16)
            sumtest_audio_b64 = self._encode_audio_to_base64(sumtest_audio, sumtest_sr)

            # Validate model with parallel intelligence tests
            pr_info("Validating model access...")
            try:
                self._validation_results = self._run_validation_tests(test_audio_silence_b64, sumtest_audio_b64)
                return self._validation_results['overall_success']

            except self.litellm_exceptions.AuthenticationError:
                pr_crit("Model validation failed: authentication error")
                pr_err(f"Check your API key environment variable for this provider")
                return False
            except self.litellm_exceptions.NotFoundError:
                pr_crit("Model validation failed: model not found")
                pr_err(f"Verify the model name and provider prefix are correct")
                return False
            except self.litellm_exceptions.RateLimitError:
                pr_crit("Model validation failed: rate limit exceeded")
                return False
            except Exception as e:
                pr_crit("Model validation failed")
                pr_err(f"Error validating model '{self.config.model_id}': {e}")
                return False

        except ImportError:
            pr_alert("litellm library not found. Install with: pip install litellm")
            return False
        except Exception as e:
            pr_err(f"Error initializing LiteLLM: {e}")
            return False

    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._initialized and self.litellm is not None

    @property
    def validation_results(self) -> Optional[dict]:
        """Get validation test results from initialization."""
        return self._validation_results

    def _encode_audio_to_base64(self, audio_np: np.ndarray, sample_rate: int) -> str:
        """Encode audio numpy array to base64 WAV string."""
        wav_bytes_io = io.BytesIO()
        sf.write(wav_bytes_io, audio_np, sample_rate, format='WAV', subtype='PCM_16')
        wav_bytes = wav_bytes_io.getvalue()
        wav_bytes_io.close()
        return base64.b64encode(wav_bytes).decode('utf-8')

    def _run_validation_tests(self, test_audio_silence_b64: str, sumtest_audio_b64: str):
        """
        Run parallel validation tests with intelligence checking.

        Args:
            test_audio_silence_b64: Base64 encoded silent audio
            sumtest_audio_b64: Base64 encoded sumtest.wav audio

        Returns:
            dict: Validation results with test outcomes
        """
        import concurrent.futures
        import re

        text_error = None
        text_response = None
        audio_error = None
        audio_response = None
        combined1_error = None
        combined1_response = None
        combined2_error = None
        combined2_response = None

        def test_text():
            completion_params = {
                "model": self.model_without_route,
                "messages": [{"role": "user", "content": "1 + 1 compute exactly only provide answer"}],
                "max_tokens": 512,
                "stream": False
            }
            if self.route:
                route_params = self.mapper.map_route_to_completion_params(self.route)
                completion_params.update(route_params)
            if self.config.api_key:
                completion_params["api_key"] = self.config.api_key
            return self.litellm.completion(**completion_params)

        def test_audio():
            audio_content = self.mapper.map_audio_params(sumtest_audio_b64, "wav")
            completion_params = {
                "model": self.model_without_route,
                "messages": [{"role": "user", "content": [audio_content]}],
                "max_tokens": 512,
                "stream": False
            }
            if self.route:
                route_params = self.mapper.map_route_to_completion_params(self.route)
                completion_params.update(route_params)
            if self.config.api_key:
                completion_params["api_key"] = self.config.api_key
            return self.litellm.completion(**completion_params)

        def test_combined1_text_with_silence():
            audio_content = self.mapper.map_audio_params(test_audio_silence_b64, "wav")
            completion_params = {
                "model": self.model_without_route,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "1 + 1 compute exactly only provide answer"},
                    audio_content
                ]}],
                "max_tokens": 512,
                "stream": False
            }
            if self.route:
                route_params = self.mapper.map_route_to_completion_params(self.route)
                completion_params.update(route_params)
            if self.config.api_key:
                completion_params["api_key"] = self.config.api_key
            return self.litellm.completion(**completion_params)

        def test_combined2_audio_with_prompt():
            audio_content = self.mapper.map_audio_params(sumtest_audio_b64, "wav")
            completion_params = {
                "model": self.model_without_route,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "compute value"},
                    audio_content
                ]}],
                "max_tokens": 512,
                "stream": False
            }
            if self.route:
                route_params = self.mapper.map_route_to_completion_params(self.route)
                completion_params.update(route_params)
            if self.config.api_key:
                completion_params["api_key"] = self.config.api_key
            return self.litellm.completion(**completion_params)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            text_future = executor.submit(test_text)
            audio_future = executor.submit(test_audio)
            combined1_future = executor.submit(test_combined1_text_with_silence)
            combined2_future = executor.submit(test_combined2_audio_with_prompt)

            try:
                text_result = text_future.result()
                text_response = text_result.choices[0].message.content
                pr_debug(f"text_response raw: {repr(text_response)}")
                if text_response is None:
                    text_response = ""
                else:
                    text_response = text_response.strip()
                pr_debug(f"text_response stripped: {repr(text_response)}")
            except Exception as e:
                text_error = e
                pr_debug(f"text_error: {e}")

            try:
                audio_result = audio_future.result()
                audio_response = audio_result.choices[0].message.content
                pr_debug(f"audio_response raw: {repr(audio_response)}")

                # Check for reasoning_content if main content is empty/minimal
                if audio_response is None or len(audio_response.strip()) < 3:
                    reasoning = getattr(audio_result.choices[0].message, 'reasoning_content', None)
                    if reasoning:
                        pr_debug(f"audio reasoning_content found: {repr(reasoning[:100])}")
                        audio_response = reasoning

                if audio_response is None:
                    audio_response = ""
                else:
                    audio_response = audio_response.strip()
                pr_debug(f"audio_response stripped: {repr(audio_response)}")
            except Exception as e:
                audio_error = e
                pr_debug(f"audio_error: {e}")

            try:
                combined1_result = combined1_future.result()
                combined1_response = combined1_result.choices[0].message.content
                pr_debug(f"combined1_response raw: {repr(combined1_response)}")
                if combined1_response is None:
                    combined1_response = ""
                else:
                    combined1_response = combined1_response.strip()
                pr_debug(f"combined1_response stripped: {repr(combined1_response)}")
            except Exception as e:
                combined1_error = e
                pr_debug(f"combined1_error: {e}")

            try:
                combined2_result = combined2_future.result()
                combined2_response = combined2_result.choices[0].message.content
                pr_debug(f"combined2_response raw: {repr(combined2_response)}")
                if combined2_response is None:
                    combined2_response = ""
                else:
                    combined2_response = combined2_response.strip()
                pr_debug(f"combined2_response stripped: {repr(combined2_response)}")
            except Exception as e:
                combined2_error = e
                pr_debug(f"combined2_error: {e}")

        def check_intelligence(response):
            if response and re.search(r'\b2\b|two', response, re.IGNORECASE):
                return True
            return False

        # For raw audio source, allow text-only failure if audio tests pass
        audio_only_passed = (audio_error is None and combined1_error is None and combined2_error is None)
        all_passed = (text_error is None and audio_error is None and
                     combined1_error is None and combined2_error is None)

        # Determine overall success
        overall_success = all_passed or (self.config.audio_source == 'raw' and audio_only_passed)

        # Helper to format response for display (replace newlines with space)
        def format_response(resp):
            if resp:
                return resp.replace('\n', ' ').replace('\r', ' ')
            return resp

        if text_error is None:
            pr_info("Text validation: passed")
            if check_intelligence(text_response):
                pr_info(f"Text intelligence test: passed - Got: {format_response(text_response)}")
            else:
                pr_warn(f"Text intelligence test: unexpected - Expected '2' but got: {format_response(text_response)}")
        else:
            pr_err(f"Text validation failed: {text_error}")

        if audio_error is None:
            pr_info("Audio validation: passed")
            if check_intelligence(audio_response):
                pr_info(f"Audio intelligence test: passed - Got: {format_response(audio_response)}")
            else:
                pr_warn(f"Audio intelligence test: unexpected - Expected '2' but got: {format_response(audio_response)}")
        else:
            pr_err(f"Audio validation failed: {audio_error}")

        if combined1_error is None:
            pr_info("Combined (text+silence) validation: passed")
            if check_intelligence(combined1_response):
                pr_info(f"Combined (text+silence) intelligence test: passed - Got: {format_response(combined1_response)}")
            else:
                pr_warn(f"Combined (text+silence) intelligence test: unexpected - Expected '2' but got: {format_response(combined1_response)}")
        else:
            pr_err(f"Combined (text+silence) validation failed: {combined1_error}")

        if combined2_error is None:
            pr_info("Combined (audio+prompt) validation: passed")
            if check_intelligence(combined2_response):
                pr_info(f"Combined (audio+prompt) intelligence test: passed - Got: {format_response(combined2_response)}")
            else:
                pr_warn(f"Combined (audio+prompt) intelligence test: unexpected - Expected '2' but got: {format_response(combined2_response)}")
        else:
            pr_err(f"Combined (audio+prompt) validation failed: {combined2_error}")

        # Print overall validation result
        if overall_success:
            pr_info("Model validation complete: passed")
            self._initialized = True
        else:
            pr_err("Model validation failed")

        # Return structured results dict
        return {
            'overall_success': overall_success,
            'text_passed': text_error is None,
            'text_error': str(text_error) if text_error else None,
            'text_response': text_response,
            'audio_passed': audio_error is None,
            'audio_error': str(audio_error) if audio_error else None,
            'audio_response': audio_response,
            'combined1_passed': combined1_error is None,
            'combined1_error': str(combined1_error) if combined1_error else None,
            'combined1_response': combined1_response,
            'combined2_passed': combined2_error is None,
            'combined2_error': str(combined2_error) if combined2_error else None,
            'combined2_response': combined2_response
        }

    def _generate_response(self, instructions: str, context, audio_data, text_data):
        """
        Generate response stream from LiteLLM API.

        Builds messages in LiteLLM format and yields response chunks.
        """
        # System message: Static instructions (cached for some providers)
        system_content = {"type": "text", "text": instructions}

        if self.provider == 'anthropic':
            system_content["cache_control"] = {"type": "ephemeral"}

        system_message = {
            "role": "system",
            "content": [system_content]
        }

        # Build user content based on input type
        if audio_data is not None:
            # Audio input
            audio_b64 = self._encode_audio_to_base64(audio_data, context.sample_rate)
            user_content = []

            context_text = self._build_context_text(context)
            user_content.append({"type": "text", "text": context_text})
            user_content.append({"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}})
        else:
            # Text input (from transcription)
            context_text = self._build_context_text(context)
            input_text = self._build_text_input_explanation(text_data)
            user_content = f"{context_text}\n\n{input_text}"

        messages = [
            system_message,
            {"role": "user", "content": user_content}
        ]

        # Display what's being sent
        self._display_user_content(user_content)

        # Build completion params
        completion_params = {
            "model": self.model_without_route,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": self.config.temperature,
            "timeout": self.config.connection_timeout
        }

        if self.route:
            route_params = self.mapper.map_route_to_completion_params(self.route)
            completion_params.update(route_params)

        if self.config.max_tokens is not None:
            completion_params["max_tokens"] = self.config.max_tokens

        if self.config.api_key:
            completion_params["api_key"] = self.config.api_key

        # Map reasoning parameters via provider-specific mapper
        if self.mapper.supports_reasoning(self.model_without_route):
            reasoning_params = self.mapper.map_reasoning_params(
                self.config.enable_reasoning,
                self.config.thinking_budget
            )
            completion_params.update(reasoning_params)

        try:
            response = self.litellm.completion(**completion_params)
            for chunk in response:
                yield chunk

        except self.litellm_exceptions.InternalServerError as e:
            pr_err("Dictation API error: Internal error encountered")
            pr_err("This is a transient error from the API provider")
            pr_err(f"Error details: {str(e)}")
            raise

    def _extract_text(self, chunk) -> Optional[str]:
        """Extract text content from LiteLLM response chunk."""
        delta = chunk.choices[0].delta
        if delta.content is not None:
            return delta.content
        return None

    def _extract_reasoning(self, chunk) -> Optional[str]:
        """Extract reasoning content from LiteLLM response chunk."""
        delta = chunk.choices[0].delta
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
            return delta.reasoning_content
        return None

    def _extract_thinking(self, chunk) -> Optional[list]:
        """Extract thinking blocks from LiteLLM response chunk."""
        delta = chunk.choices[0].delta
        if hasattr(delta, 'thinking_blocks') and delta.thinking_blocks is not None:
            return delta.thinking_blocks
        return None

    def _extract_usage(self, chunk) -> Optional[dict]:
        """Extract usage statistics from LiteLLM response chunk."""
        if hasattr(chunk, 'usage') and chunk.usage is not None:
            return chunk.usage
        return None

    def _display_user_content(self, user_content):
        """Display user content being sent to model."""
        pr_debug("=" * 60)
        pr_debug("SENDING TO MODEL:")

        # Handle list format (audio transcription)
        if isinstance(user_content, list):
            for content_block in user_content:
                if content_block["type"] == "text":
                    pr_debug(content_block["text"])
                elif content_block["type"] == "input_audio":
                    pr_debug("Audio: audio_data.wav (base64)")
        # Handle string format (text transcription)
        else:
            pr_debug(user_content)

        pr_debug("-" * 60)

    def _on_response_complete(self, usage_data, last_chunk) -> None:
        """Display cache stats after response completes."""
        if usage_data:
            self._display_cache_stats(usage_data, completion_response=last_chunk)

    def _display_cache_stats(self, usage_data, completion_response=None) -> None:
        """Display cache statistics and cost from usage data."""
        if not self.config.debug_enabled:
            return

        pr_debug("-" * 60)
        pr_debug("USAGE STATISTICS:")

        # Standard token counts
        if hasattr(usage_data, 'prompt_tokens'):
            pr_debug(f"  Prompt tokens: {usage_data.prompt_tokens}")
        if hasattr(usage_data, 'completion_tokens'):
            pr_debug(f"  Completion tokens: {usage_data.completion_tokens}")
        if hasattr(usage_data, 'total_tokens'):
            pr_debug(f"  Total tokens: {usage_data.total_tokens}")

        # Anthropic-specific cache fields
        if hasattr(usage_data, 'cache_creation_input_tokens') and usage_data.cache_creation_input_tokens:
            pr_debug(f"  Cache creation tokens: {usage_data.cache_creation_input_tokens} (Anthropic: written to cache)")

        if hasattr(usage_data, 'cache_read_input_tokens') and usage_data.cache_read_input_tokens:
            pr_debug(f"  Cache read tokens: {usage_data.cache_read_input_tokens} (Anthropic: read from cache)")

        # DeepSeek-specific cache fields
        if hasattr(usage_data, 'prompt_cache_hit_tokens') and usage_data.prompt_cache_hit_tokens:
            pr_debug(f"  Cache hit tokens: {usage_data.prompt_cache_hit_tokens} (DeepSeek: cache hits)")

        if hasattr(usage_data, 'prompt_cache_miss_tokens') and usage_data.prompt_cache_miss_tokens:
            pr_debug(f"  Cache miss tokens: {usage_data.prompt_cache_miss_tokens} (DeepSeek: cache misses)")

        # OpenAI/Gemini format: prompt_tokens_details
        if hasattr(usage_data, 'prompt_tokens_details') and usage_data.prompt_tokens_details:
            details = usage_data.prompt_tokens_details

            # Show audio tokens if present
            if hasattr(details, 'audio_tokens') and details.audio_tokens:
                pr_debug(f"  Audio tokens: {details.audio_tokens}")

            # Show text tokens if present
            if hasattr(details, 'text_tokens') and details.text_tokens:
                pr_debug(f"  Text tokens: {details.text_tokens}")

            # Show cached tokens (None = no caching, 0 = cache warming, >0 = cache hit)
            if hasattr(details, 'cached_tokens'):
                if details.cached_tokens is None:
                    pr_debug("  Cached tokens: None (no implicit caching detected)")
                elif details.cached_tokens == 0:
                    pr_debug("  Cached tokens: 0 (cache warming - first request)")
                else:
                    pr_debug(f"  Cached tokens: {details.cached_tokens} (cache hit)")

        # Completion token details
        if hasattr(usage_data, 'completion_tokens_details') and usage_data.completion_tokens_details:
            details = usage_data.completion_tokens_details

            # Show reasoning tokens if present (extended thinking)
            if hasattr(details, 'reasoning_tokens') and details.reasoning_tokens:
                pr_debug(f"  Reasoning tokens: {details.reasoning_tokens} (extended thinking)")

        # Gemini-specific: cached_content_token_count (alternative field)
        if hasattr(usage_data, 'cached_content_token_count') and usage_data.cached_content_token_count:
            pr_debug(f"  Cached content tokens: {usage_data.cached_content_token_count} (Gemini: implicit cache)")

        # Calculate and display cost
        if completion_response:
            try:
                current_cost = self.litellm.completion_cost(completion_response=completion_response)
                self.total_cost += current_cost
                pr_debug("COST:")
                pr_debug(f"  Current request: ${current_cost:.6f}")
                pr_debug(f"  Total session: ${self.total_cost:.6f}")
            except Exception as e:
                pr_debug(f"COST: Unable to calculate ({str(e)})")

        pr_debug("-" * 60)
