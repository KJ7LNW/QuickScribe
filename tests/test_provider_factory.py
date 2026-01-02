"""Test provider factory selection logic."""
import pytest
from providers.provider_factory import create_provider
from providers.registry import extract_provider
from providers.litellm_provider import LiteLLMProvider
from providers.huggingface_provider import HuggingFaceProvider
from config_manager import ConfigManager


class MockAudioProcessor:
    """Mock audio processor for testing."""
    pass


def test_extract_provider_openai():
    """Test provider extraction from openai model_id."""
    assert extract_provider('openai/gpt-4') == 'openai'


def test_extract_provider_huggingface():
    """Test provider extraction from huggingface model_id."""
    assert extract_provider('huggingface/Qwen/Qwen2.5-0.5B') == 'huggingface'


def test_extract_provider_empty():
    """Test provider extraction from empty model_id."""
    assert extract_provider('') == ''
    assert extract_provider(None) == ''


def test_factory_creates_litellm_provider():
    """Test factory creates LiteLLMProvider for cloud APIs."""
    config = ConfigManager()
    config.model_id = 'openai/gpt-4'

    provider = create_provider(config, MockAudioProcessor())

    assert isinstance(provider, LiteLLMProvider)
    assert provider.provider == 'openai'


def test_factory_creates_huggingface_provider():
    """Test factory creates HuggingFaceProvider for huggingface/ prefix."""
    config = ConfigManager()
    config.model_id = 'huggingface/Qwen/Qwen2.5-0.5B'

    provider = create_provider(config, MockAudioProcessor())

    assert isinstance(provider, HuggingFaceProvider)
    assert provider.provider == 'huggingface'


def test_factory_default_to_litellm():
    """Test factory defaults to LiteLLMProvider for unknown providers."""
    config = ConfigManager()
    config.model_id = 'groq/llama-3.3-70b-versatile'

    provider = create_provider(config, MockAudioProcessor())

    assert isinstance(provider, LiteLLMProvider)
    assert provider.provider == 'groq'




class MockContext:
    """Mock conversation context for testing."""

    def __init__(self, xml_markup=None, compiled_text=None):
        self.xml_markup = xml_markup
        self.compiled_text = compiled_text


def test_huggingface_provider_extract_text():
    """Test HuggingFaceProvider text extraction from plain chunks."""
    config = ConfigManager()
    config.model_id = 'huggingface/test/model'

    provider = HuggingFaceProvider(config, MockAudioProcessor())

    assert provider.extract_text("test chunk") == "test chunk"
    assert provider.extract_text("") == ""


def test_huggingface_provider_extract_reasoning_returns_none():
    """Test HuggingFaceProvider returns None for reasoning."""
    config = ConfigManager()
    config.model_id = 'huggingface/test/model'

    provider = HuggingFaceProvider(config, MockAudioProcessor())

    assert provider.extract_reasoning("any chunk") is None


def test_huggingface_provider_extract_usage_returns_none():
    """Test HuggingFaceProvider returns None for usage."""
    config = ConfigManager()
    config.model_id = 'huggingface/test/model'

    provider = HuggingFaceProvider(config, MockAudioProcessor())

    assert provider.extract_usage("any chunk") is None


def test_huggingface_provider_display_user_content():
    """Test HuggingFaceProvider display content formatting."""
    config = ConfigManager()
    config.model_id = 'huggingface/test/model'

    provider = HuggingFaceProvider(config, MockAudioProcessor())

    content = {
        'messages': [
            {"role": "system", "content": "System instructions"},
            {"role": "user", "content": "User input text"}
        ],
        'prompt': "formatted prompt string",
        'tokens': 42
    }

    # Should not raise - display is debug output only
    provider._display_user_content(content)
