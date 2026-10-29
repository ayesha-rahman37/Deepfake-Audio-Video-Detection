"""Temporal video window construction using half-open intervals."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TemporalSegment:
    """A video interval ``[start, end)`` and its chronological index."""

    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def segment_video(
    duration: float,
    window_duration: float = 5.0,
    overlap: float = 0.0,
) -> list[TemporalSegment]:
    """Cover ``[0, duration)`` with ordered windows, retaining a partial tail.

    Overlap is explicit and defaults to zero. A zero-duration video yields no
    segments; negative/non-finite duration and invalid window settings fail.
    """
    values = (duration, window_duration, overlap)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Duration, window duration, and overlap must be finite.")
    if duration < 0:
        raise ValueError("Duration cannot be negative.")
    if window_duration <= 0:
        raise ValueError("Window duration must be greater than zero.")
    if overlap < 0 or overlap >= window_duration:
        raise ValueError("Overlap must be at least zero and less than window duration.")
    if duration == 0:
        return []

    step = window_duration - overlap
    segments: list[TemporalSegment] = []
    start = 0.0
    while start < duration:
        end = min(start + window_duration, duration)
        segments.append(TemporalSegment(len(segments), start, end))
        start += step
    return segments
