"""Audio signal processing utilities for QuickScribe."""

import numpy as np
from typing import Iterator
from concurrent.futures import ThreadPoolExecutor

try:
    import pyrubberband as pyrb
except ImportError:
    pyrb = None


def _stretch_single(
    audio_data: np.ndarray,
    sample_rate: int,
    speed_factor: float
) -> tuple[int, np.ndarray]:
    """
    Stretch audio at single speed factor.

    Args:
        audio_data: Original audio data
        sample_rate: Audio sample rate in Hz
        speed_factor: Speed multiplier (e.g., 0.80, 0.90, 1.0)

    Returns:
        Tuple of (speed_pct, stretched_audio)
    """
    speed_pct = int(speed_factor * 100)

    if speed_factor == 1.0:
        return (speed_pct, audio_data)

    stretched_audio = pyrb.time_stretch(audio_data, sample_rate, speed_factor)
    return (speed_pct, stretched_audio)


def stretch_audio_at_speeds(
    audio_data: np.ndarray,
    sample_rate: int,
    speed_factors: list[float]
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Generate audio variants at different playback speeds.

    Args:
        audio_data: Original audio data as numpy array
        sample_rate: Audio sample rate in Hz
        speed_factors: List of speed multipliers (e.g., [0.80, 0.90, 1.0])

    Yields:
        Tuple of (speed_pct, stretched_audio) where speed_pct is the
        integer percentage (e.g., 80, 90, 100)

    Raises:
        ImportError: If pyrubberband is not installed
    """
    if pyrb is None:
        raise ImportError(
            "pyrubberband library not installed. "
            "Install with: pip install pyrubberband"
        )

    for speed_factor in speed_factors:
        speed_pct = int(speed_factor * 100)

        # Optimization: avoid processing for original speed
        if speed_factor == 1.0:
            yield (speed_pct, audio_data)
        else:
            stretched_audio = pyrb.time_stretch(audio_data, sample_rate, speed_factor)
            yield (speed_pct, stretched_audio)


def stretch_audio_parallel(
    audio_data: np.ndarray,
    sample_rate: int,
    speed_factors: list[float]
) -> list[tuple[int, np.ndarray]]:
    """
    Process all speed variants in parallel using ThreadPoolExecutor.

    Args:
        audio_data: Original audio data as numpy array
        sample_rate: Audio sample rate in Hz
        speed_factors: List of speed multipliers (e.g., [0.80, 0.90, 1.0])

    Returns:
        List of (speed_pct, stretched_audio) tuples in original speed_factors order

    Raises:
        ImportError: If pyrubberband is not installed
    """
    if pyrb is None:
        raise ImportError(
            "pyrubberband library not installed. "
            "Install with: pip install pyrubberband"
        )

    with ThreadPoolExecutor(max_workers=len(speed_factors)) as executor:
        futures = [
            executor.submit(_stretch_single, audio_data, sample_rate, speed_factor)
            for speed_factor in speed_factors
        ]

        results = [future.result() for future in futures]

    return results
