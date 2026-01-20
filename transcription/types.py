"""Data types for transcription pipeline."""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class TranscriptionInput:
    """Input data for a single transcription operation."""
    audio_data: np.ndarray
    sample_rate: int
    speed_pct: int


@dataclass
class TranscriptionResult:
    """Result from a single transcription operation."""
    text: str
    speed_pct: int
    duration_ms: int
    error: Optional[str]
