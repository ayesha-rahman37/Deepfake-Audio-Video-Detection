
from pathlib import Path
import csv

import numpy as np
import soundfile as sf
import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = PROJECT_ROOT / "data" / "raw" / "In the name of love.mp4"
AUDIO_PATH = (
    PROJECT_ROOT / "data" / "processed" / "audio"
    / "In the name of love.wav"
)

VIDEO_WINDOWS_DIR = PROJECT_ROOT / "data" / "processed" / "frames"
AUDIO_WINDOWS_DIR = PROJECT_ROOT / "data" / "processed" / "audio_windows"

OUTPUT_PATH = PROJECT_ROOT / "data" / "metadata" / "metadata.csv"
WINDOW_SECONDS = 5.0


def get_video_duration():
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps <= 0:
        raise RuntimeError("Invalid video FPS.")

    return frame_count / fps


def get_audio_duration():
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")

    return sf.info(str(AUDIO_PATH)).duration


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    if not VIDEO_WINDOWS_DIR.exists():
        raise FileNotFoundError(
            f"Video windows not found: {VIDEO_WINDOWS_DIR}"
        )

    if not AUDIO_WINDOWS_DIR.exists():
        raise FileNotFoundError(
            f"Audio windows not found: {AUDIO_WINDOWS_DIR}"
        )

    video_duration = get_video_duration()
    audio_duration = get_audio_duration()
    duration = min(video_duration, audio_duration)

    video_windows = {
        path.name: path
        for path in VIDEO_WINDOWS_DIR.glob("window_*")
        if path.is_dir()
    }

    audio_windows = {
        path.stem: path
        for path in AUDIO_WINDOWS_DIR.glob("window_*.npy")
    }

    matching_names = sorted(
        set(video_windows) & set(audio_windows)
    )

    if not matching_names:
        raise RuntimeError("No matching audio-video windows found.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for window_name in matching_names:
        try:
            window_index = int(window_name.split("_")[-1])
        except ValueError:
            print(f"Skipping unexpected window name: {window_name}")
            continue

        start_time = window_index * WINDOW_SECONDS

        if start_time >= duration:
            continue

        end_time = min(start_time + WINDOW_SECONDS, duration)

        frame_paths = sorted(
            video_windows[window_name].glob("frame_*.jpg")
        )

        audio_feature = np.load(audio_windows[window_name])

        if not frame_paths:
            print(f"Skipping {window_name}: no video frames")
            continue

        if audio_feature.ndim != 2 or audio_feature.size == 0:
            print(f"Skipping {window_name}: invalid audio features")
            continue

        rows.append({
            "sample_id": window_name,
            "video_path": str(video_windows[window_name].relative_to(PROJECT_ROOT)),
            "audio_feature_path": str(audio_windows[window_name].relative_to(PROJECT_ROOT)),
            "start_seconds": round(start_time, 3),
            "end_seconds": round(end_time, 3),
            "frame_count": len(frame_paths),
            "audio_feature_frames": int(audio_feature.shape[1]),
            "label": "unknown",
            "split": "unassigned",
        })

    if not rows:
        raise RuntimeError("No valid metadata rows were generated.")

    fieldnames = [
        "sample_id",
        "video_path",
        "audio_feature_path",
        "start_seconds",
        "end_seconds",
        "frame_count",
        "audio_feature_frames",
        "label",
        "split",
    ]

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nMetadata generation successful!")
    print(f"Samples written: {len(rows)}")
    print(f"Output CSV: {OUTPUT_PATH.resolve()}")
    print("\nFirst three rows:")

    for row in rows[:3]:
        print(row)


if __name__ == "__main__":
    main()
