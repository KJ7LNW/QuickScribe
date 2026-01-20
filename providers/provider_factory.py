"""
Provider factory for instantiating LLM backends.

Uses registry for provider selection based on model_id prefix.
"""
from providers.registry import get_implementation, get_default, extract_provider, ProviderCapability


def create_provider(config):
    """
    Factory for provider instantiation.

    Args:
        config: ConfigManager instance

    Returns:
        Provider instance (LiteLLMProvider, HuggingFaceProvider, etc.)
    """
    provider_name = extract_provider(config.model_id)
    impl = get_implementation(provider_name, ProviderCapability.LLM)

    if impl:
        return impl(config)

    return get_default(ProviderCapability.LLM)(config)
