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


def transcribe_file(audio_file: str, config: ConfigManager) -> str:
    """
    Transcribe audio file using configuration from config.

    Args:
        audio_file: Path to audio file
        config: Fully populated ConfigManager instance

    Returns:
        Transcribed text
    """
    audio_data, file_sample_rate = load_audio_file(audio_file)

    target_sample_rate = config.sample_rate

    if file_sample_rate != target_sample_rate:
        pr_info(f"Resampling from {file_sample_rate} Hz to {target_sample_rate} Hz")
        num_samples = int(len(audio_data) * target_sample_rate / file_sample_rate)
        audio_data = signal.resample(audio_data, num_samples).astype('int16')
    else:
        target_sample_rate = file_sample_rate

    pr_info(f"Initializing transcription with model: {config.transcription_model}")

    transcription_source = get_transcription_source(config)

    # Apply system instruction overrides before initialization
    if config.sys_instructions is not None:
        transcription_source.instructions = config.sys_instructions

    if config.sys_append is not None:
        transcription_source.instructions += "\n" + config.sys_append

    if not transcription_source.initialize():
        raise RuntimeError("Failed to initialize transcription source")

    transcription_input = TranscriptionInput(
        audio_data=audio_data,
        sample_rate=target_sample_rate,
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
        help="Transcription model specification (e.g., 'vosk/model-path', 'openai/whisper-1', 'gemini/gemini-2.0-flash')"
    )

    parser.add_argument(
        "--sample-rate", "-r",
        type=int,
        default=16000,
        help="Target sample rate for transcription model"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: stdout)"
    )

    parser.add_argument(
        "--debug", "-D",
        action="count",
        default=0,
        help="Enable debug output (-D app debug, -DD also enables litellm debug)"
    )

    parser.add_argument(
        "--enable-reasoning", "-R",
        choices=["none", "low", "medium", "high"],
        default="low",
        help="Reasoning effort level sent to model"
    )

    parser.add_argument(
        "--thinking-budget", "-B",
        type=int,
        default=128,
        help="Token budget for model reasoning/thinking"
    )

    parser.add_argument(
        "--http-timeout", "-H",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds for API requests"
    )

    parser.add_argument(
        "--key", "-k",
        type=str,
        default=None,
        help="API key override for the transcription provider"
    )

    parser.add_argument(
        "--sys", "-s",
        type=str,
        default=None,
        help="Replace system instructions with this string instead of reading the markdown file"
    )

    parser.add_argument(
        "--sys-append", "-a",
        type=str,
        default=None,
        help="Append this text to system instructions before sending (applied after --sys if both given)"
    )

    args = parser.parse_args()

    debug_level = args.debug
    if debug_level >= 1:
        set_log_level(PR_DEBUG)
    else:
        set_log_level(PR_ERR)

    config = ConfigManager()
    config.transcription_model = args.transcription_model
    config.sample_rate = args.sample_rate
    config.audio_source = "transcribe"
    config.debug_enabled = debug_level >= 1
    config.litellm_debug = debug_level >= 2
    config.enable_reasoning = args.enable_reasoning
    config.thinking_budget = args.thinking_budget
    config.http_timeout = args.http_timeout
    config.api_key = args.key
    config.sys_instructions = args.sys
    config.sys_append = args.sys_append

    try:
        text = transcribe_file(args.audio_file, config)

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
