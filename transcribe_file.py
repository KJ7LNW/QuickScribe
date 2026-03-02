#!/usr/bin/env python3
"""
Simple audio file transcription utility using QuickScribe libraries.
"""
import sys
import argparse
import numpy as np
import soundfile as sf
from scipy import signal
from config_manager import ConfigManager
from transcription.factory import get_transcription_source
from transcription.types import TranscriptionInput
from lib.pr_log import pr_info, pr_err, set_log_level, PR_EMERG, PR_ERR, PR_DEBUG


def load_audio_file(file_path: str) -> tuple[np.ndarray, int]:
    """
    Load audio file and return audio data and sample rate.

    Args:
        file_path: Path to audio file

    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        audio_data, sample_rate = sf.read(file_path, dtype='int16')

        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1).astype('int16')

        pr_info(f"Loaded audio: {file_path}")
        pr_info(f"Sample rate: {sample_rate} Hz, Duration: {len(audio_data)/sample_rate:.2f}s")

        return audio_data, sample_rate

    except Exception as e:
        pr_err(f"Error loading audio file: {e}")
        raise


def transcribe_file(audio_file: str, transcription_model: str, target_sample_rate: int = 16000) -> str:
    """
    Transcribe audio file using specified model.

    Args:
        audio_file: Path to audio file
        transcription_model: Model specification (e.g., "vosk/model", "openai/whisper-1")
        target_sample_rate: Target sample rate for transcription model

    Returns:
        Transcribed text
    """
    audio_data, file_sample_rate = load_audio_file(audio_file)

    if file_sample_rate != target_sample_rate:
        pr_info(f"Resampling from {file_sample_rate} Hz to {target_sample_rate} Hz")
        num_samples = int(len(audio_data) * target_sample_rate / file_sample_rate)
        audio_data = signal.resample(audio_data, num_samples).astype('int16')
        sample_rate = target_sample_rate
    else:
        sample_rate = file_sample_rate

    config = ConfigManager()
    config.transcription_model = transcription_model
    config.sample_rate = sample_rate
    config.audio_source = "transcribe"
    config.debug_enabled = False

    pr_info(f"Initializing transcription with model: {transcription_model}")

    transcription_source = get_transcription_source(config)

    if not transcription_source.initialize():
        raise RuntimeError("Failed to initialize transcription source")

    transcription_input = TranscriptionInput(
        audio_data=audio_data,
        sample_rate=sample_rate,
        speed_pct=100
    )

    pr_info("Starting transcription...")
    result = transcription_source.transcribe_audio_data(transcription_input)

    if result.error:
        pr_err(f"Transcription error: {result.error}")
        return ""

    pr_info(f"Transcription completed in {result.duration_ms}ms")

    transcription_source._cleanup()

    return result.text


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio file to text using QuickScribe libraries",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "audio_file",
        help="Path to audio file (WAV, MP3, etc.)"
    )

    parser.add_argument(
        "--transcription-model", "-T",
        type=str,
        default="vosk/vosk-model-small-en-us-0.15",
        help="Transcription model specification (e.g., 'vosk/model-path', 'openai/whisper-1', 'huggingface/facebook/wav2vec2-base-960h')"
    )

    parser.add_argument(
        "--sample-rate", "-r",
        type=int,
        default=16000,
        help="Target sample rate for transcription model (default: 16000 Hz)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: stdout)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output to stderr"
    )

    args = parser.parse_args()

    if args.verbose:
        set_log_level(PR_DEBUG)
    else:
        set_log_level(PR_ERR)

    try:
        text = transcribe_file(args.audio_file, args.transcription_model, args.sample_rate)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(text)
                f.write('\n')
        else:
            print(text)

        return 0

    except Exception as e:
        pr_err(f"Transcription failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
