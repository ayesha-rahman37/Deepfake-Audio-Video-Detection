import subprocess

import imageio_ffmpeg
import numpy as np
import pytest

from preprocessing import frame_sampler, video_reader
from preprocessing.temporal_segmenter import segment_video
from preprocessing.video_reader import VideoReader


class FakeCapture:
    def __init__(self, frames, timestamps, fps=10.0, reported_count=None, opened=True):
        self.frames = frames
        self.timestamps = timestamps
        self.fps = fps
        self.reported_count = (
            len(frames) if reported_count is None else reported_count
        )
        self.opened = opened
        self.position = 0
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, prop):
        if prop == video_reader.cv2.CAP_PROP_FPS:
            return self.fps
        if prop == video_reader.cv2.CAP_PROP_FRAME_COUNT:
            return self.reported_count
        if prop == video_reader.cv2.CAP_PROP_FRAME_WIDTH:
            return self.frames[0].shape[1] if self.frames else 0
        if prop == video_reader.cv2.CAP_PROP_FRAME_HEIGHT:
            return self.frames[0].shape[0] if self.frames else 0
        if prop == video_reader.cv2.CAP_PROP_POS_MSEC:
            if self.position == 0:
                return 0
            return self.timestamps[self.position - 1] * 1000
        return 0

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame.copy() if frame is not None else None

    def release(self):
        self.released = True


def install_capture(monkeypatch, frames, timestamps, **kwargs):
    instances = []

    def factory(path):
        assert path
        capture = FakeCapture(frames, timestamps, **kwargs)
        instances.append(capture)
        return capture

    monkeypatch.setattr(video_reader.cv2, "VideoCapture", factory)
    return instances


def make_frames(count):
    return [
        np.full((8, 12, 3), fill_value=index, dtype=np.uint8)
        for index in range(count)
    ]


def test_segment_video_covers_partial_tail_without_gaps():
    segments = segment_video(12)
    assert [(part.start, part.end) for part in segments] == [
        (0.0, 5.0),
        (5.0, 10.0),
        (10.0, 12.0),
    ]


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        (2.5, [(0.0, 2.5)]),
        (5.0, [(0.0, 5.0)]),
        (5.01, [(0.0, 5.0), (5.0, 5.01)]),
        (0.0, []),
    ],
)
def test_segment_video_duration_cases(duration, expected):
    assert [(part.start, part.end) for part in segment_video(duration)] == expected


@pytest.mark.parametrize(
    ("duration", "window", "overlap"),
    [(-1, 5, 0), (1, 0, 0), (1, 5, -1), (1, 5, 5), (float("nan"), 5, 0)],
)
def test_segment_video_rejects_invalid_values(duration, window, overlap):
    with pytest.raises(ValueError):
        segment_video(duration, window, overlap)


def test_video_reader_yields_ordered_full_rgb_frames_and_metadata(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.mp4"
    path.write_bytes(b"synthetic")
    frames = make_frames(4)
    timestamps = [0.0, 0.1, 0.2, 0.3]
    captures = install_capture(monkeypatch, frames, timestamps)

    reader = VideoReader(path)
    decoded = list(reader.iter_frames())

    assert [frame.index for frame in decoded] == [0, 1, 2, 3]
    assert [frame.timestamp for frame in decoded] == timestamps
    assert all(frame.timestamp_source == "decoder_timestamp" for frame in decoded)
    assert decoded[2].image.shape == (8, 12, 3)
    assert np.array_equal(decoded[2].image, frames[2][:, :, ::-1])
    assert reader.metadata.frame_count_decoded == 4
    assert reader.metadata.duration == pytest.approx(0.4)
    assert captures[0].released


def test_video_reader_falls_back_when_timestamps_do_not_advance(tmp_path, monkeypatch):
    path = tmp_path / "fallback.mp4"
    path.write_bytes(b"synthetic")
    install_capture(monkeypatch, make_frames(3), [0, 0, 0], fps=10.0)

    reader = VideoReader(path)
    decoded = list(reader.iter_frames())

    assert [frame.timestamp for frame in decoded] == pytest.approx([0.0, 0.1, 0.2])
    assert decoded[1].timestamp_source == "fps_estimate"
    assert any("estimated from reported FPS" in item for item in reader.metadata.warnings)


def test_video_reader_normalizes_nonzero_decoder_timestamp_origin(
    tmp_path, monkeypatch
):
    path = tmp_path / "nonzero-start.mp4"
    path.write_bytes(b"synthetic")
    install_capture(
        monkeypatch, make_frames(3), [50.0, 50.1, 50.2], fps=10.0
    )

    reader = VideoReader(path)
    frames = list(reader.iter_frames())

    assert [frame.timestamp for frame in frames] == pytest.approx([0.0, 0.1, 0.2])
    assert reader.metadata.duration == pytest.approx(0.3)


def test_video_reader_reports_metadata_disagreement(tmp_path, monkeypatch):
    path = tmp_path / "mismatch.mp4"
    path.write_bytes(b"synthetic")
    install_capture(
        monkeypatch, make_frames(3), [0, 0.1, 0.2], reported_count=5
    )

    reader = VideoReader(path)
    list(reader.iter_frames())

    assert any("Reported 5 frames but decoded 3" in item for item in reader.metadata.warnings)


def test_video_reader_rejects_missing_unreadable_and_empty_videos(
    tmp_path, monkeypatch
):
    missing = VideoReader(tmp_path / "missing.mp4")
    with pytest.raises(FileNotFoundError):
        list(missing.iter_frames())

    unreadable_path = tmp_path / "unreadable.mp4"
    unreadable_path.write_bytes(b"x")
    captures = install_capture(monkeypatch, [], [], opened=False)
    with pytest.raises(RuntimeError, match="Cannot open video"):
        list(VideoReader(unreadable_path).iter_frames())
    assert captures[0].released

    install_capture(monkeypatch, [], [])
    with pytest.raises(RuntimeError, match="no decodable frames"):
        list(VideoReader(unreadable_path).iter_frames())


def test_sample_video_uniformly_samples_and_deduplicates(tmp_path, monkeypatch):
    path = tmp_path / "twelve-seconds.mp4"
    path.write_bytes(b"synthetic")
    count = 120
    timestamps = [i / 10 for i in range(count)]
    install_capture(monkeypatch, make_frames(count), timestamps)

    sampled = frame_sampler.sample_video(path)

    assert [(item.segment.start, item.segment.end) for item in sampled.segments] == [
        (0.0, 5.0),
        (5.0, 10.0),
        (10.0, 12.0),
    ]
    assert [len(item.frames) for item in sampled.segments] == [16, 16, 16]
    for item in sampled.segments:
        assert [frame.timestamp for frame in item.frames] == sorted(
            frame.timestamp for frame in item.frames
        )
        assert len({frame.frame_index for frame in item.frames}) == len(item.frames)
        assert all(
            item.segment.start <= frame.timestamp < item.segment.end
            for frame in item.frames
        )
        assert all(
            abs(frame.timestamp - frame.target_timestamp) <= 0.051
            for frame in item.frames
        )
        assert all(frame.image.shape == (8, 12, 3) for frame in item.frames)


def test_sample_video_keeps_only_available_unique_frames(tmp_path, monkeypatch):
    path = tmp_path / "short.mp4"
    path.write_bytes(b"synthetic")
    install_capture(monkeypatch, make_frames(3), [0.0, 0.1, 0.2], fps=10.0)

    sampled = frame_sampler.sample_video(
        path, frames_per_window=16, window_duration=5.0
    )

    assert len(sampled.segments) == 1
    assert len(sampled.segments[0].frames) == 3
    assert any("3 unique frames for 16 targets" in item for item in sampled.warnings)


def test_video_reader_skips_malformed_frame_but_preserves_source_indices(
    tmp_path, monkeypatch
):
    path = tmp_path / "corrupt-frame.mp4"
    path.write_bytes(b"synthetic")
    frames = [make_frames(1)[0], None, make_frames(1)[0]]
    install_capture(monkeypatch, frames, [0.0, 0.1, 0.2], reported_count=3)

    reader = VideoReader(path)
    decoded = list(reader.iter_frames())

    assert [frame.index for frame in decoded] == [0, 2]
    assert reader.metadata.duration == pytest.approx(0.3)
    assert any("frame 1 was malformed" in item for item in reader.metadata.warnings)
    assert any("Reported 3 frames but decoded 2" in item for item in reader.metadata.warnings)


def test_sampler_uses_irregular_decoder_timestamps(tmp_path, monkeypatch):
    path = tmp_path / "variable-intervals.mp4"
    path.write_bytes(b"synthetic")
    timestamps = [0.0]
    for index in range(50):
        timestamps.append(timestamps[-1] + (0.08 if index % 2 == 0 else 0.12))
    install_capture(monkeypatch, make_frames(len(timestamps)), timestamps, fps=10.0)

    sampled = frame_sampler.sample_video(path)

    first_segment = sampled.segments[0]
    assert first_segment.segment.end == pytest.approx(5.0)
    assert len(first_segment.frames) == 16
    assert all(
        abs(frame.timestamp - frame.target_timestamp) <= 0.061
        for frame in first_segment.frames
    )


def test_video_reader_decodes_a_real_synthetic_video(tmp_path):
    path = tmp_path / "synthetic.mp4"
    result = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1.2:size=64x48:rate=10",
            "-an",
            "-c:v",
            "mpeg4",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    reader = VideoReader(path)
    frames = list(reader.iter_frames())

    assert len(frames) >= 10
    assert all(frame.image.shape == (48, 64, 3) for frame in frames)
    assert all(
        current.timestamp > previous.timestamp
        for previous, current in zip(frames, frames[1:])
    )
    assert reader.metadata.duration == pytest.approx(1.2, abs=0.2)
