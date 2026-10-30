"""Uniform, full-frame sampling over sequentially decoded video."""

from __future__ import annotations

from dataclasses import dataclass, field
from bisect import bisect_right
from pathlib import Path
import warnings

import cv2
import numpy as np
import yaml

from preprocessing.temporal_segmenter import TemporalSegment, segment_video
from preprocessing.video_reader import VideoFrame, VideoMetadata, VideoReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class SampledFrame:
    """A selected original full frame associated with one target position."""

    image: np.ndarray
    timestamp: float
    frame_index: int
    timestamp_source: str
    segment_index: int
    segment_start: float
    segment_end: float
    sample_position: int
    target_timestamp: float


@dataclass(frozen=True)
class SampledSegment:
    """One temporal segment and its unique selected source frames."""

    segment: TemporalSegment
    frames: tuple[SampledFrame, ...]


@dataclass
class SampledVideo:
    """Sampling results with decoder metadata and all warnings."""

    metadata: VideoMetadata
    segments: list[SampledSegment]
    warnings: list[str] = field(default_factory=list)


def load_config() -> dict:
    """Load project configuration used by the legacy convenience function."""
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid project configuration: {CONFIG_PATH}")
    return config


def _sample_targets(segment: TemporalSegment, count: int) -> list[float]:
    """Return bin-center targets inside the half-open segment."""
    return [
        segment.start + (position + 0.5) * segment.duration / count
        for position in range(count)
    ]


def sample_video(
    video_path: str | Path,
    frames_per_window: int = 16,
    window_duration: float = 5.0,
    overlap: float = 0.0,
) -> SampledVideo:
    """Return up to ``frames_per_window`` unique RGB frames per segment.

    The video is decoded sequentially twice: the first pass measures its
    decoded duration, and the second chooses the closest valid frame for each
    evenly spaced bin-center target. This avoids trusting inaccurate container
    duration/FPS metadata and retains only target candidates in memory.
    If several targets resolve to one source frame, it is returned only once;
    the segment may therefore contain fewer than the requested number.
    """
    if isinstance(frames_per_window, bool) or frames_per_window < 1:
        raise ValueError("frames_per_window must be a positive integer.")
    if int(frames_per_window) != frames_per_window:
        raise ValueError("frames_per_window must be a positive integer.")

    first_pass = VideoReader(video_path)
    for _ in first_pass.iter_frames():
        pass
    metadata = first_pass.metadata
    if metadata is None or metadata.duration is None or metadata.duration <= 0:
        raise RuntimeError(
            f"Could not determine a positive decoded duration for {video_path}."
        )

    segments = segment_video(metadata.duration, window_duration, overlap)
    targets = [_sample_targets(segment, frames_per_window) for segment in segments]
    segment_starts = [segment.start for segment in segments]
    best: list[list[tuple[float, VideoFrame] | None]] = [
        [None] * frames_per_window for _ in segments
    ]
    second_pass = VideoReader(video_path)
    for frame in second_pass.iter_frames():
        segment_index = bisect_right(segment_starts, frame.timestamp) - 1
        while segment_index >= 0 and segments[segment_index].end > frame.timestamp:
            segment = segments[segment_index]
            if segment.start <= frame.timestamp < segment.end:
                for position, target in enumerate(targets[segment_index]):
                    distance = abs(frame.timestamp - target)
                    previous = best[segment_index][position]
                    if previous is None or distance < previous[0]:
                        best[segment_index][position] = (distance, frame)
            segment_index -= 1

    sampled_segments: list[SampledSegment] = []
    warnings_list = list(dict.fromkeys(metadata.warnings))
    if second_pass.metadata is not None:
        warnings_list.extend(
            warning
            for warning in second_pass.metadata.warnings
            if warning not in warnings_list
        )
        if (
            second_pass.metadata.duration is not None
            and abs(second_pass.metadata.duration - metadata.duration) > 0.1
        ):
            warnings_list.append(
                "The decoded duration changed between the sampling passes."
            )

    for segment_index, segment in enumerate(segments):
        selected: dict[int, SampledFrame] = {}
        for position, candidate in enumerate(best[segment_index]):
            if candidate is None:
                continue
            frame = candidate[1]
            if frame.index in selected:
                continue
            selected[frame.index] = SampledFrame(
                image=frame.image,
                timestamp=frame.timestamp,
                frame_index=frame.index,
                timestamp_source=frame.timestamp_source,
                segment_index=segment.index,
                segment_start=segment.start,
                segment_end=segment.end,
                sample_position=position,
                target_timestamp=targets[segment_index][position],
            )
        selected_frames = tuple(
            sorted(selected.values(), key=lambda item: (item.timestamp, item.frame_index))
        )
        if len(selected_frames) < frames_per_window:
            warnings_list.append(
                f"Segment {segment.index} produced {len(selected_frames)} unique "
                f"frames for {frames_per_window} targets."
            )
        sampled_segments.append(SampledSegment(segment, selected_frames))

    return SampledVideo(metadata, sampled_segments, warnings_list)


def sample_frames(video_path: str | Path, output_dir: str | Path) -> int:
    """Compatibility wrapper saving sampled RGB frames as JPEGs.

    The existing project entry point and return value are retained. The new
    :func:`sample_video` API exposes timestamps, indices, segment boundaries,
    metadata, and warnings for training/inference code.
    """
    config = load_config()
    data_config = config.get("data", {})
    sampled = sample_video(
        video_path,
        frames_per_window=data_config.get("frames_per_window", 16),
        window_duration=data_config.get("window_duration", 5.0),
        overlap=data_config.get("window_overlap", 0.0),
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    for item in sampled.segments:
        window_dir = output_path / f"window_{item.segment.index:04d}"
        window_dir.mkdir(parents=True, exist_ok=True)
        for sample in item.frames:
            bgr = cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR)
            frame_path = window_dir / f"frame_{sample.sample_position:02d}.jpg"
            if not cv2.imwrite(str(frame_path), bgr):
                raise RuntimeError(f"Could not write sampled frame: {frame_path}")
            saved_count += 1

    for warning in sampled.warnings:
        warnings.warn(warning, RuntimeWarning, stacklevel=2)
    return saved_count


if __name__ == "__main__":
    sample_frames(
        PROJECT_ROOT / "data" / "raw" / "In the name of love.mp4",
        PROJECT_ROOT / "data" / "processed" / "frames",
    )
