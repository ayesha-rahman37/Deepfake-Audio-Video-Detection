
import sys
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg


def inspect_video(video_path):
    video_path = Path(video_path)

    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return

    print("\n" + "=" * 50)
    print("VIDEO INSPECTION REPORT")
    print("=" * 50)
    print(f"File: {video_path.name}")
    print(f"Size: {video_path.stat().st_size / (1024 * 1024):.2f} MB")

    # Inspect video frames and metadata
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: OpenCV could not open this video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps > 0 else 0

    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {frame_count}")
    print(f"Duration: {duration:.2f} seconds")

    success, frame = cap.read()
    print(f"First frame readable: {'Yes' if success else 'No'}")

    cap.release()

    # Check whether an audio stream exists
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    result = subprocess.run(
        [ffmpeg, "-i", str(video_path), "-hide_banner"],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30
    )

    output = result.stderr

    has_audio = any(
        "Audio:" in line for line in output.splitlines()
    )

    has_video = any(
        "Video:" in line for line in output.splitlines()
    )

    print(f"Video stream detected: {'Yes' if has_video else 'No'}")
    print(f"Audio stream detected: {'Yes' if has_audio else 'No'}")
    print("=" * 50)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            'Usage: python preprocessing/inspect_dataset.py '
            '"path_to_video.mp4"'
        )
        sys.exit(1)

    inspect_video(sys.argv[1])
