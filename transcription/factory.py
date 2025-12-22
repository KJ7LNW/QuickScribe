"""Factory for creating transcription audio sources."""

from providers.registry import get_implementation, get_default, extract_provider, ProviderCapability
from transcription.implementations.huggingface.ctc import HuggingFaceCTCTranscriptionAudioSource
from transcription.implementations.huggingface.seq2seq import (
    WhisperTranscriptionAudioSource,
    Speech2TextTranscriptionAudioSource
)
from transcription.implementations.huggingface.nemo_tdt import NeMoTDTTranscriptionAudioSource


_HUGGINGFACE_IMPLEMENTATIONS = {
    'ctc': HuggingFaceCTCTranscriptionAudioSource,
    'whisper': WhisperTranscriptionAudioSource,
    'speech2text': Speech2TextTranscriptionAudioSource,
    'nemo_tdt': NeMoTDTTranscriptionAudioSource,
}


def get_transcription_source(config):
    """
    Create transcription audio source based on model specification.

    Args:
        config: Configuration object with transcription_model attribute

    Returns:
        TranscriptionAudioSource instance

    Raises:
        ValueError: If provider not supported or model format invalid
    """
    transcription_model = config.transcription_model

    if '/' not in transcription_model:
        raise ValueError(
            f"Invalid transcription model format: '{transcription_model}'. "
            "Expected format: provider/model-identifier"
        )

    provider = extract_provider(transcription_model)

    # Check for registered transcription provider
    impl = get_implementation(provider, ProviderCapability.TRANSCRIPTION)
    if impl:
        return impl(config, transcription_model)

    # Special case: HuggingFace requires architecture detection
    if provider == 'huggingface':
        from transcription.implementations.huggingface.model_loader import load_huggingface_model

        # Extract model identifier (remove provider prefix)
        model_identifier = transcription_model.split('/', 1)[1]
        model, processor, architecture = load_huggingface_model(
            model_identifier,
            cache_dir=None,
            force_download=False,
            local_files_only=False
        )

        implementation_class = _HUGGINGFACE_IMPLEMENTATIONS.get(architecture)
        if implementation_class is None:
            raise ValueError(
                f"Unsupported HuggingFace architecture: '{architecture}'. "
                f"Supported architectures: {', '.join(_HUGGINGFACE_IMPLEMENTATIONS.keys())}"
            )

        return implementation_class(config, model, processor)

    # Fall back to LiteLLM transcription for unregistered providers
    return get_default(ProviderCapability.TRANSCRIPTION)(config)
