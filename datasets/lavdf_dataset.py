"""LAV-DF dataset loader for aligned full-frame video/audio windows.

This version preserves the existing return keys used by the training scripts
and handles very short trailing windows caused by small metadata-duration
mismatches. It does not silently synthesize audio when extraction fails.
"""

import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import librosa
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(
    os.environ.get("LAVDF_ROOT", r"D:\LAV-DF")
).expanduser()
METADATA_CSV = PROJECT_ROOT / "data" / "metadata" / "lavdf_metadata.csv"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# Only inspect the actual container duration when metadata would create a very
# short trailing window. This avoids opening every video when indexing a split.
SHORT_FINAL_WINDOW_PROBE_SECONDS = 0.5


def _read_video_info(video_path: Path) -> tuple[float, float, int]:
    """Return (duration_seconds, fps, frame_count) from the video container."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not np.isfinite(fps) or fps <= 0 or frame_count <= 0:
            raise RuntimeError(
                f"Invalid video properties for {video_path}: "
                f"fps={fps}, frame_count={frame_count}"
            )

        return frame_count / fps, fps, frame_count
    finally:
        cap.release()


def _parse_fake_periods(raw: str, sample_id: str) -> list[tuple[float, float]]:
    """Validate and parse temporal forgery annotations."""
    try:
        periods: Any = json.loads(raw or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid fake_periods JSON for sample {sample_id}: {exc}"
        ) from exc

    if not isinstance(periods, list):
        raise ValueError(
            f"fake_periods must be a list for sample {sample_id}; "
            f"got {type(periods).__name__}"
        )

    parsed: list[tuple[float, float]] = []
    for i, period in enumerate(periods):
        if not isinstance(period, (list, tuple)) or len(period) < 2:
            raise ValueError(
                f"Invalid fake period #{i} for sample {sample_id}: {period!r}"
            )

        start, end = float(period[0]), float(period[1])
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError(
                f"Invalid fake period #{i} for sample {sample_id}: {period!r}"
            )
        parsed.append((start, end))

    return parsed


class LAVDFDataset(Dataset):
    """Load aligned LAV-DF full-frame video and audio windows.

    The label returned as ``label`` is window-level: 1 when that window
    overlaps any annotated fake period, otherwise 0. Audio and video are
    sampled from the same start/end timestamps.
    """

    def __init__(self, split: str = "train"):
        if split not in {"train", "dev", "test"}:
            raise ValueError("split must be 'train', 'dev', or 'test'")

        if not METADATA_CSV.is_file():
            raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")
        if not CONFIG_PATH.is_file():
            raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        settings = config.get("data", {})
        self.window_seconds = float(settings["window_duration"])
        self.frames_per_window = int(settings["frames_per_window"])
        self.image_size = int(settings["image_size"])
        self.sample_rate = int(settings["audio_sample_rate"])

        if self.window_seconds <= 0:
            raise ValueError("data.window_duration must be greater than zero")
        if self.frames_per_window < 1:
            raise ValueError("data.frames_per_window must be at least 1")
        if self.image_size < 1 or self.sample_rate < 1:
            raise ValueError("image_size and audio_sample_rate must be positive")

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        self.samples: list[dict[str, Any]] = []
        missing_videos = 0
        invalid_rows = 0
        duration_adjustments = 0
        short_tail_corrections = 0

        with METADATA_CSV.open("r", encoding="utf-8-sig", newline="") as file:
            records = csv.DictReader(file)

            required_columns = {
                "split",
                "duration",
                "video_path",
                "sample_id",
                "fake_periods",
            }
            available_columns = set(records.fieldnames or [])
            missing_columns = required_columns - available_columns
            if missing_columns:
                raise ValueError(
                    "Metadata CSV is missing required columns: "
                    + ", ".join(sorted(missing_columns))
                )

            for record in records:
                if (record.get("split") or "").strip() != split:
                    continue

                sample_id = (record.get("sample_id") or "").strip()
                relative_video_path = (record.get("video_path") or "").strip()
                if not relative_video_path:
                    invalid_rows += 1
                    continue

                try:
                    duration = float(record["duration"])
                except (TypeError, ValueError):
                    invalid_rows += 1
                    continue

                if not np.isfinite(duration) or duration <= 0:
                    invalid_rows += 1
                    continue

                video_path = (DATASET_ROOT / relative_video_path).resolve()
                if not video_path.is_file():
                    missing_videos += 1
                    continue

                fake_periods = _parse_fake_periods(
                    record.get("fake_periods", "[]"), sample_id or relative_video_path
                )

                # When the metadata creates a tiny last window, inspect the real
                # video duration so an empty/out-of-range window is not indexed.
                remainder = duration % self.window_seconds
                if (
                    1e-6 < remainder <= SHORT_FINAL_WINDOW_PROBE_SECONDS
                ):
                    actual_duration, fps, _ = _read_video_info(video_path)
                    duration_adjustments += int(
                        abs(actual_duration - duration) > 1e-3
                    )
                    final_window_start = duration - remainder
                    one_frame_seconds = 1.0 / fps

                    if actual_duration <= final_window_start + one_frame_seconds:
                        # Drop a trailing sliver no longer than one frame, while
                        # retaining a legitimate earlier partial window.
                        duration = min(actual_duration, final_window_start)
                        short_tail_corrections += 1
                    else:
                        # Trust the smaller boundary if metadata is slightly longer.
                        duration = min(duration, actual_duration)

                if duration <= 0:
                    invalid_rows += 1
                    continue

                start = 0.0
                while start < duration - 1e-6:
                    end = min(start + self.window_seconds, duration)
                    if end <= start:
                        break

                    is_fake = any(
                        period_start < end and period_end > start
                        for period_start, period_end in fake_periods
                    )

                    self.samples.append(
                        {
                            "video_path": video_path,
                            "sample_id": sample_id or video_path.stem,
                            "start": start,
                            "end": end,
                            "label": int(is_fake),
                        }
                    )
                    start = end

        if not self.samples:
            raise RuntimeError(
                f"No usable samples found for split={split}. "
                f"Check LAVDF_ROOT={DATASET_ROOT} and metadata paths."
            )

        self.index_report = {
            "split": split,
            "windows": len(self.samples),
            "missing_videos": missing_videos,
            "invalid_rows": invalid_rows,
            "duration_adjustments": duration_adjustments,
            "short_tail_corrections": short_tail_corrections,
        }

        print(
            f"LAV-DF {split}: {len(self.samples)} windows from local videos\n"
            f"Missing videos skipped: {missing_videos}; "
            f"invalid metadata rows skipped: {invalid_rows}; "
            f"duration adjustments: {duration_adjustments}; "
            f"short-tail corrections: {short_tail_corrections}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_video(self, video_path: Path, start: float, end: float) -> torch.Tensor:
        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")

            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not np.isfinite(fps) or fps <= 0 or frame_count <= 0:
                raise RuntimeError(
                    f"Invalid video properties for {video_path}: "
                    f"fps={fps}, frame_count={frame_count}"
                )

            actual_duration = frame_count / fps
            clip_start = max(0.0, start)
            clip_end = min(end, actual_duration)
            if clip_end <= clip_start + 1e-6:
                raise RuntimeError(
                    f"Empty video window [{start:.3f}, {end:.3f}] for "
                    f"{video_path}; actual duration={actual_duration:.3f}s. "
                    "Rebuild/check the metadata duration before training."
                )

            # Keep timestamps strictly before the last frame boundary.
            last_valid_time = max(clip_start, clip_end - 1.0 / fps)
            times = np.linspace(clip_start, last_valid_time, self.frames_per_window)
            frames: list[torch.Tensor] = []
            last_image: Image.Image | None = None

            for timestamp in times:
                frame_index = min(max(int(timestamp * fps), 0), frame_count - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = cap.read()

                if success and frame is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    last_image = Image.fromarray(rgb)
                elif last_image is None:
                    raise RuntimeError(
                        f"Cannot decode frame at {timestamp:.3f}s in {video_path}"
                    )
                # If a later frame fails, repeat the last decoded full frame.
                frames.append(self.transform(last_image))

            result = torch.stack(frames, dim=0)
            expected_shape = (self.frames_per_window, 3, self.image_size, self.image_size)
            if tuple(result.shape) != expected_shape:
                raise RuntimeError(
                    f"Unexpected video tensor shape {tuple(result.shape)}; "
                    f"expected {expected_shape} for {video_path}"
                )
            return result
        finally:
            cap.release()

    def _load_audio(self, video_path: Path, start: float, end: float) -> torch.Tensor:
        segment_duration = end - start
        if segment_duration <= 0:
            raise ValueError(
                f"Invalid audio window [{start:.3f}, {end:.3f}] for {video_path}"
            )

        with tempfile.TemporaryDirectory(prefix="lavdf_audio_") as temp_dir:
            wav_path = Path(temp_dir) / "segment.wav"
            command = [
                self.ffmpeg,
                "-y",
                "-v",
                "error",
                "-ss",
                f"{start:.6f}",
                "-i",
                str(video_path),
                "-t",
                f"{segment_duration:.6f}",
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(self.sample_rate),
                "-c:a",
                "pcm_s16le",
                str(wav_path),
            ]

            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Audio extraction timed out for {video_path} "
                    f"at {start:.3f}s"
                ) from exc

            if result.returncode != 0 or not wav_path.is_file():
                details = (result.stderr or "No FFmpeg error details")[-500:]
                raise RuntimeError(
                    f"Audio extraction failed for {video_path} "
                    f"[{start:.3f}, {end:.3f}]: {details}"
                )

            audio, _ = librosa.load(
                str(wav_path),
                sr=self.sample_rate,
                mono=True,
            )

        if audio.size == 0:
            raise RuntimeError(
                f"No decodable audio in {video_path} for "
                f"[{start:.3f}, {end:.3f}]. Missing audio is not silently replaced."
            )

        # Fixed-length waveform input. Only the short end of a real final window
        # is zero-padded; audio is never substituted for an absent audio stream.
        target_samples = round(self.window_seconds * self.sample_rate)
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))
        else:
            audio = audio[:target_samples]

        mel_power = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=64,
            power=2.0,
        )
        log_mel = librosa.power_to_db(mel_power, ref=np.max).astype(np.float32)

        if log_mel.ndim != 2 or log_mel.shape[0] != 64:
            raise RuntimeError(
                f"Unexpected Log-Mel shape {log_mel.shape} for {video_path}"
            )
        if not np.isfinite(log_mel).all():
            raise RuntimeError(
                f"Non-finite Log-Mel values for {video_path} "
                f"[{start:.3f}, {end:.3f}]"
            )
        return torch.from_numpy(log_mel)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]

        video = self._load_video(
            sample["video_path"], sample["start"], sample["end"]
        )
        audio = self._load_audio(
            sample["video_path"], sample["start"], sample["end"]
        )

        return {
            "sample_id": sample["sample_id"],
            "start_seconds": sample["start"],
            "end_seconds": sample["end"],
            "video": video,  # (frames_per_window, 3, image_size, image_size)
            "audio": audio,  # (64, time_frames)
            "label": torch.tensor(sample["label"], dtype=torch.long),
        }


if __name__ == "__main__":
    dataset = LAVDFDataset(split="train")
    sample = dataset[0]

    print("\nLAV-DF sample test successful!")
    print("Video:", tuple(sample["video"].shape))
    print("Audio:", tuple(sample["audio"].shape))
    print("Label:", sample["label"].item())
    print(
        "Window:",
        round(sample["start_seconds"], 3),
        "-",
        round(sample["end_seconds"], 3),
    )

    # Regression check for the four video records examined in the duration audit.
    audited_ids = {"092507", "092509", "114497", "114499"}
    by_id: dict[str, list[int]] = {}
    for idx, item in enumerate(dataset.samples):
        if item["sample_id"] in audited_ids:
            by_id.setdefault(item["sample_id"], []).append(idx)

    for sample_id in sorted(audited_ids):
        indices = by_id.get(sample_id, [])
        if not indices:
            print(f"Boundary regression: {sample_id} not present in train split")
            continue
        boundary_sample = dataset[indices[-1]]
        print(
            f"Boundary check {sample_id}: PASS; "
            f"window={boundary_sample['start_seconds']:.3f}-"
            f"{boundary_sample['end_seconds']:.3f}s; "
            f"video={tuple(boundary_sample['video'].shape)}; "
            f"audio={tuple(boundary_sample['audio'].shape)}"
        )

    print("Index report:", dataset.index_report)
