
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = PROJECT_ROOT / "data" / "raw" / "In the name of love.mp4"
AUDIO_PATH = (
    PROJECT_ROOT / "data" / "processed" / "audio"
    / "In the name of love.wav"
)
VIDEO_FRAMES_DIR = PROJECT_ROOT / "data" / "processed" / "frames"
AUDIO_WINDOWS_DIR = PROJECT_ROOT / "data" / "processed" / "audio_windows"

WINDOW_SECONDS = 5.0
TOLERANCE_SECONDS = 0.15


def inspect_video():
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps <= 0:
        raise RuntimeError("Invalid video FPS.")

    return frame_count / fps


def inspect_audio():
    info = sf.info(str(AUDIO_PATH))
    return info.duration


def count_saved_video_windows():
    if not VIDEO_FRAMES_DIR.exists():
        return 0

    return sum(
        1 for p in VIDEO_FRAMES_DIR.glob("window_*") if p.is_dir()
    )


def count_saved_audio_windows():
    if not AUDIO_WINDOWS_DIR.exists():
        return 0

    return sum(1 for p in AUDIO_WINDOWS_DIR.glob("window_*.npy"))


def main():
    for path in (VIDEO_PATH, AUDIO_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    video_duration = inspect_video()
    audio_duration = inspect_audio()

    video_windows = count_saved_video_windows()
    audio_windows = count_saved_audio_windows()

    expected_video_windows = int(
        np.ceil(video_duration / WINDOW_SECONDS)
    )
    expected_audio_windows = int(
        np.ceil(audio_duration / WINDOW_SECONDS)
    )

    print("=" * 50)
    print("AUDIO-VIDEO ALIGNMENT CHECK")
    print("=" * 50)
    print(f"Video duration: {video_duration:.3f} seconds")
    print(f"Audio duration: {audio_duration:.3f} seconds")
    print(f"Duration difference: {abs(video_duration - audio_duration):.3f} seconds")
    print(f"Expected video windows: {expected_video_windows}")
    print(f"Saved video windows: {video_windows}")
    print(f"Expected audio windows: {expected_audio_windows}")
    print(f"Saved audio windows: {audio_windows}")

    if abs(video_duration - audio_duration) <= TOLERANCE_SECONDS:
        print("Duration check: PASS")
    else:
        print("Duration check: REVIEW NEEDED")

    if video_windows == audio_windows == expected_video_windows:
        print("Window count check: PASS")
    else:
        print("Window count check: REVIEW NEEDED")

    print("=" * 50)


if __name__ == "__main__":
    main()
