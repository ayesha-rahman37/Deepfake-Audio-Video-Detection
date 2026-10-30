"""Sequential video decoding with explicit timestamp and metadata provenance."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from pathlib import Path
from statistics import median
from typing import Iterator, Literal

import cv2
import numpy as np


TimestampSource = Literal["decoder_timestamp", "fps_estimate"]


@dataclass
class VideoMetadata:
    """Container metadata reconciled with observations from decoded frames.

    ``duration`` is the decoded duration when timestamps permit it. It is
    estimated from the last frame timestamp plus the median observed frame
    interval (or ``1 / fps``); it is not a claim about exact container duration.
    """

    source_path: str
    fps_reported: float | None
    frame_count_reported: int | None
    width: int | None
    height: int | None
    duration_reported: float | None
    duration: float | None = None
    duration_source: str | None = None
    frame_count_decoded: int = 0
    timestamp_source: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class VideoFrame:
    """A decoded RGB frame and its original stream position."""

    index: int
    timestamp: float
    timestamp_source: TimestampSource
    image: np.ndarray


class VideoReader:
    """Decode a video sequentially without retaining prior frames.

    OpenCV's decoder-position property is accepted as a source timestamp only
    when it is finite and strictly increasing. Values are made relative to the
    first frame. OpenCV does not guarantee that this property is the exact
    presentation timestamp for every backend/container; it is therefore
    labeled ``decoder_timestamp`` rather than asserted to be source PTS.
    Missing or non-monotonic values fall back to ``frame_index / reported_fps``
    and are marked estimated. That fallback assumes constant frame rate and
    is not exact for variable-frame-rate media.
    """

    def __init__(self, video_path: str | Path):
        self.video_path = Path(video_path)
        self.metadata: VideoMetadata | None = None

    def iter_frames(self) -> Iterator[VideoFrame]:
        """Yield every decoded frame in chronological decoder order."""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        if not self.video_path.is_file():
            raise ValueError(f"Video path is not a file: {self.video_path}")

        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        fps_value = float(capture.get(cv2.CAP_PROP_FPS))
        count_value = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width_value = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_value = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = fps_value if math.isfinite(fps_value) and fps_value > 0 else None
        frame_count = (
            int(round(count_value))
            if math.isfinite(count_value) and count_value > 0
            else None
        )
        duration_reported = (
            frame_count / fps if frame_count is not None and fps is not None else None
        )
        self.metadata = VideoMetadata(
            source_path=str(self.video_path.resolve()),
            fps_reported=fps,
            frame_count_reported=frame_count,
            width=(
                int(width_value)
                if math.isfinite(width_value) and width_value > 0
                else None
            ),
            height=(
                int(height_value)
                if math.isfinite(height_value) and height_value > 0
                else None
            ),
            duration_reported=duration_reported,
        )
        meta = self.metadata
        intervals: deque[float] = deque(maxlen=2048)
        last_timestamp: float | None = None
        last_frame_index: int | None = None
        sources: set[TimestampSource] = set()
        previous_timestamp: float | None = None
        previous_reported_timestamp: float | None = None
        timestamp_origin: float | None = None
        stream_index = 0

        try:
            while True:
                ok, bgr_frame = capture.read()
                if not ok:
                    break
                index = stream_index
                stream_index += 1
                if bgr_frame is None or bgr_frame.ndim != 3 or bgr_frame.shape[2] != 3:
                    meta.warnings.append(
                        f"Decoded stream frame {index} was malformed and skipped."
                    )
                    continue

                if meta.width is None or meta.height is None:
                    meta.height, meta.width = bgr_frame.shape[:2]
                elif bgr_frame.shape[:2] != (meta.height, meta.width):
                    meta.warnings.append(
                        f"Frame {index} dimensions {bgr_frame.shape[1]}x"
                        f"{bgr_frame.shape[0]} differ from reported video dimensions "
                        f"{meta.width}x{meta.height}."
                    )

                raw_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                raw_timestamp = raw_ms / 1000.0
                has_decoder_timestamp = (
                    math.isfinite(raw_timestamp)
                    and raw_timestamp >= 0
                    and (
                        previous_reported_timestamp is None
                        or raw_timestamp > previous_reported_timestamp + 1e-9
                    )
                )
                if has_decoder_timestamp:
                    if timestamp_origin is None:
                        timestamp_origin = raw_timestamp
                    timestamp = raw_timestamp - timestamp_origin
                    if (
                        previous_timestamp is not None
                        and timestamp <= previous_timestamp + 1e-9
                    ):
                        has_decoder_timestamp = False
                    else:
                        source: TimestampSource = "decoder_timestamp"
                        previous_reported_timestamp = raw_timestamp
                if not has_decoder_timestamp and fps is not None:
                    timestamp = index / fps
                    source = "fps_estimate"
                    if previous_timestamp is not None and timestamp <= previous_timestamp:
                        timestamp = previous_timestamp + 1.0 / fps
                    meta.warnings.append(
                        f"Frame {index} has no reliable increasing decoder-position "
                        "timestamp; timestamp estimated from reported FPS."
                    )
                elif not has_decoder_timestamp and previous_timestamp is not None:
                    raise RuntimeError(
                        "Cannot timestamp decoded frames: decoder timestamps "
                        "are unavailable and reported FPS is invalid."
                    )
                elif not has_decoder_timestamp:
                    raise RuntimeError(
                        "Cannot timestamp the first decoded frame: decoder "
                        "timestamp and reported FPS are unavailable."
                    )

                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                if (
                    last_timestamp is not None
                    and last_frame_index is not None
                    and timestamp > last_timestamp
                ):
                    skipped_span = index - last_frame_index
                    intervals.append((timestamp - last_timestamp) / skipped_span)
                sources.add(source)
                meta.frame_count_decoded += 1
                previous_timestamp = timestamp
                last_timestamp = timestamp
                last_frame_index = index
                yield VideoFrame(index, timestamp, source, rgb_frame)
        finally:
            capture.release()

        if meta.frame_count_decoded == 0:
            raise RuntimeError(f"Video contains no decodable frames: {self.video_path}")

        if frame_count is not None and frame_count != meta.frame_count_decoded:
            meta.warnings.append(
                f"Reported {frame_count} frames but decoded "
                f"{meta.frame_count_decoded}."
            )
        if last_timestamp is not None:
            interval = median(intervals) if intervals else (
                1.0 / fps if fps is not None else None
            )
            if interval is not None:
                meta.duration = last_timestamp + interval
                meta.duration_source = (
                    "decoder_timestamps_plus_median_interval"
                    if sources == {"decoder_timestamp"}
                    else "decoded_timestamps_with_estimates"
                )
            elif duration_reported is not None:
                meta.duration = duration_reported
                meta.duration_source = "reported_frame_count_and_fps"
            meta.timestamp_source = (
                next(iter(sources)) if len(sources) == 1 else "mixed"
            )

        if (
            duration_reported is not None
            and meta.duration is not None
            and abs(duration_reported - meta.duration) > max(0.1, 0.05 * meta.duration)
        ):
            meta.warnings.append(
                f"Reported duration {duration_reported:.3f}s differs from decoded "
                f"duration estimate {meta.duration:.3f}s."
            )
        if meta.duration is None:
            meta.warnings.append(
                "Duration could not be established from decoded timestamps or metadata."
            )
