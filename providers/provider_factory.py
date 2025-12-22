"""
Provider factory for instantiating LLM backends.

Uses registry for provider selection based on model_id prefix.
"""
from providers.registry import get_implementation, get_default, extract_provider, ProviderCapability


def create_provider(config, audio_processor):
    """
    Factory for provider instantiation.

    Args:
        config: ConfigManager instance
        audio_processor: AudioSource instance

    Returns:
        Provider instance (LiteLLMProvider, HuggingFaceProvider, etc.)
    """
    provider_name = extract_provider(config.model_id)
    impl = get_implementation(provider_name, ProviderCapability.LLM)

    if impl:
        return impl(config, audio_processor)

    return get_default(ProviderCapability.LLM)(config, audio_processor)
