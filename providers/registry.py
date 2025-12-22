"""
Provider registry for LLM and transcription implementations.

Single point of truth for provider knowledge across the application.
"""
from enum import Enum
from typing import Optional


class ProviderCapability(Enum):
    """Provider capability types."""
    LLM = 'llm'
    TRANSCRIPTION = 'transcription'


# Registry maps provider name to capability-specific implementation
_REGISTRY: dict[str, dict[ProviderCapability, type]] = {}

# Default implementations for unregistered providers
_DEFAULTS: dict[ProviderCapability, type] = {}


def register(provider: str, capability: ProviderCapability, impl: type) -> None:
    """
    Register implementation for provider and capability.

    Args:
        provider: Provider name (e.g., 'openai', 'anthropic')
        capability: Capability type
        impl: Implementation class

    Raises:
        TypeError: If capability is not ProviderCapability enum
    """
    if not isinstance(capability, ProviderCapability):
        raise TypeError(f"capability must be ProviderCapability enum, not {type(capability)}")

    if provider not in _REGISTRY:
        _REGISTRY[provider] = {}

    _REGISTRY[provider][capability] = impl


def set_default(capability: ProviderCapability, impl: type) -> None:
    """
    Set default implementation for capability.

    Args:
        capability: Capability type
        impl: Default implementation class

    Raises:
        TypeError: If capability is not ProviderCapability enum
    """
    if not isinstance(capability, ProviderCapability):
        raise TypeError(f"capability must be ProviderCapability enum, not {type(capability)}")

    _DEFAULTS[capability] = impl


def get_implementation(provider: str, capability: ProviderCapability) -> Optional[type]:
    """
    Get implementation for provider and capability.

    Args:
        provider: Provider name
        capability: Capability type

    Returns:
        Implementation class if registered, None otherwise

    Raises:
        TypeError: If capability is not ProviderCapability enum
    """
    if not isinstance(capability, ProviderCapability):
        raise TypeError(f"capability must be ProviderCapability enum, not {type(capability)}")

    return _REGISTRY.get(provider, {}).get(capability)


def get_default(capability: ProviderCapability) -> type:
    """
    Get default implementation for capability.

    Args:
        capability: Capability type

    Returns:
        Default implementation class

    Raises:
        TypeError: If capability is not ProviderCapability enum
        KeyError: If no default registered for capability
    """
    if not isinstance(capability, ProviderCapability):
        raise TypeError(f"capability must be ProviderCapability enum, not {type(capability)}")

    if capability not in _DEFAULTS:
        raise KeyError(f"No default implementation registered for {capability}")

    return _DEFAULTS[capability]


def extract_provider(model_spec: str) -> str:
    """
    Extract provider name from model specification.

    Single point of truth for provider extraction logic.

    Args:
        model_spec: Model specification (format: "provider/model" or "provider/model@route")

    Returns:
        Provider name (lowercase) or empty string if no provider prefix
    """
    if not model_spec:
        return ''

    if '/' not in model_spec:
        return ''

    return model_spec.split('/', 1)[0].lower()


def extract_model(model_spec: str) -> str:
    """
    Extract model identifier from model specification.

    Single point of truth for model extraction logic.

    Args:
        model_spec: Model specification (format: "provider/model" or "model")

    Returns:
        Model identifier without provider prefix
    """
    if not model_spec:
        return model_spec

    if '/' not in model_spec:
        return model_spec

    return model_spec.split('/', 1)[1]


# Register LLM providers
from providers.huggingface_provider import HuggingFaceProvider
from providers.llamacpp_provider import LlamaCppProvider
from providers.none_provider import NoneProvider

register('huggingface', ProviderCapability.LLM, HuggingFaceProvider)
register('llamacpp', ProviderCapability.LLM, LlamaCppProvider)
register('gguf', ProviderCapability.LLM, LlamaCppProvider)
register('none', ProviderCapability.LLM, NoneProvider)

# Register transcription providers
from transcription.implementations.openai import OpenAITranscriptionAudioSource
from transcription.implementations.vosk import VoskTranscriptionAudioSource

register('openai', ProviderCapability.TRANSCRIPTION, OpenAITranscriptionAudioSource)
register('vosk', ProviderCapability.TRANSCRIPTION, VoskTranscriptionAudioSource)

# Set defaults for unregistered providers
from providers.litellm_provider import LiteLLMProvider
from transcription.implementations.litellm_transcription import LiteLLMTranscriptionAudioSource

set_default(ProviderCapability.LLM, LiteLLMProvider)
set_default(ProviderCapability.TRANSCRIPTION, LiteLLMTranscriptionAudioSource)
