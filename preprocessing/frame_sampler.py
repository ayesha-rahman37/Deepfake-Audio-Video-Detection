
from pathlib import Path

import cv2
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def sample_frames(video_path, output_dir):
    config = load_config()
    data_config = config["data"]

    window_seconds = data_config["window_duration"]
    frames_per_window = data_config["frames_per_window"]

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError("Invalid FPS or frame count.")

    duration = total_frames / fps
    window_index = 0
    saved_count = 0

    try:
        while window_index * window_seconds < duration - 1e-6:
            start_time = window_index * window_seconds
            end_time = min(start_time + window_seconds, duration)

            # Sample evenly across this window.
            timestamps = [
                start_time
                + (i + 0.5) * (end_time - start_time) / frames_per_window
                for i in range(frames_per_window)
            ]

            window_dir = output_dir / f"window_{window_index:04d}"
            window_dir.mkdir(parents=True, exist_ok=True)

            window_saved = 0

            for frame_index, timestamp in enumerate(timestamps):
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                success, frame = cap.read()

                if not success:
                    print(
                        f"Warning: Could not read frame at "
                        f"{timestamp:.2f}s"
                    )
                    continue

                frame_path = window_dir / f"frame_{frame_index:02d}.jpg"

                if cv2.imwrite(str(frame_path), frame):
                    window_saved += 1
                    saved_count += 1

            print(
                f"Window {window_index}: "
                f"{start_time:.2f}s - {end_time:.2f}s, "
                f"saved {window_saved}/{frames_per_window} frames"
            )

            window_index += 1

    finally:
        cap.release()

    print("\nFrame sampling completed.")
    print(f"Video duration: {duration:.2f} seconds")
    print(f"Windows processed: {window_index}")
    print(f"Total frames saved: {saved_count}")
    print(f"Output folder: {output_dir.resolve()}")

    return saved_count


if __name__ == "__main__":
    sample_frames(
        PROJECT_ROOT / "data" / "raw" / "In the name of love.mp4",
        PROJECT_ROOT / "data" / "processed" / "frames",
    )
