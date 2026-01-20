"""
Test OpenRouter provider configuration.

Verifies that @provider-name syntax maps to correct OpenRouter API structure
with fallback and latency optimization enabled.
"""
import pytest
from config_manager import ConfigManager
from providers.provider_factory import create_provider


class MockAudioProcessor:
    """Mock audio processor for testing."""
    pass


def test_openrouter_provider_configuration():
    """Test provider name maps to complete OpenRouter configuration."""
    config = ConfigManager()
    config.model_id = 'openrouter/google/gemini-2.5-flash@google-vertex'
    config.audio_source = 'raw'
    config.mode = 'dictate'

    provider = create_provider(config)

    assert provider.route == 'google-vertex'
    assert provider.provider == 'openrouter'

    result = provider.mapper.map_route_to_completion_params('google-vertex')

    assert 'extra_body' in result
    assert 'provider' in result['extra_body']

    provider_config = result['extra_body']['provider']
    assert provider_config['order'] == ['google-vertex']
    assert provider_config['allow_fallbacks'] is True
    assert provider_config['sort'] == 'latency'

    assert 'route' not in result


def test_openrouter_different_provider_names():
    """Test various provider names produce correct configuration."""
    config = ConfigManager()
    config.audio_source = 'raw'
    config.mode = 'dictate'

    provider_names = ['google-vertex', 'anthropic', 'deepinfra', 'fireworks', 'together']

    for provider_name in provider_names:
        config.model_id = f'openrouter/test/model@{provider_name}'
        provider = create_provider(config)

        result = provider.mapper.map_route_to_completion_params(provider_name)

        assert result['extra_body']['provider']['order'] == [provider_name]
        assert result['extra_body']['provider']['allow_fallbacks'] is True
        assert result['extra_body']['provider']['sort'] == 'latency'


def test_openrouter_no_route_specified():
    """Test that no route produces no provider configuration."""
    config = ConfigManager()
    config.model_id = 'openrouter/google/gemini-2.5-flash'
    config.audio_source = 'raw'
    config.mode = 'dictate'

    provider = create_provider(config)

    assert provider.route is None
    assert provider.provider == 'openrouter'
