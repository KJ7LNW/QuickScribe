"""
Provider factory for instantiating LLM backends.

Single point of truth for provider selection based on model_id prefix.
"""
from providers.huggingface_provider import HuggingFaceProvider
from providers.litellm_provider import LiteLLMProvider
from providers.llamacpp_provider import LlamaCppProvider
from providers.none_provider import NoneProvider


_PROVIDERS = {
    'huggingface': HuggingFaceProvider,
    'llamacpp': LlamaCppProvider,
    'gguf': LlamaCppProvider,
    'none': NoneProvider,
}


def _extract_provider(model_id: str) -> str:
    """Extract provider prefix from model_id."""
    if model_id and '/' in model_id:
        return model_id.split('/', 1)[0].lower()

    return ''


def create_provider(config, audio_processor):
    """
    Factory for provider instantiation.

    Args:
        config: ConfigManager instance
        audio_processor: AudioSource instance

    Returns:
        Provider instance (LiteLLMProvider or HuggingFaceProvider)
    """
    provider_name = _extract_provider(config.model_id)
    provider_class = _PROVIDERS.get(provider_name, LiteLLMProvider)

    return provider_class(config, audio_processor)
