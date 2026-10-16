
from pathlib import Path

import numpy as np
import librosa
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def segment_audio_features(feature_path, output_dir):
    feature_path = Path(feature_path)
    output_dir = Path(output_dir)

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Log-Mel feature file not found: {feature_path}"
        )

    config = load_config()
    data_config = config["data"]

    window_seconds = float(data_config["window_duration"])
    sample_rate = int(data_config["audio_sample_rate"])

    # Load the Log-Mel feature matrix: (mel_bands, time_frames)
    features = np.load(feature_path)

    if features.ndim != 2:
        raise ValueError(
            f"Expected a 2D Log-Mel matrix, got shape {features.shape}"
        )

    # This must match audio_features.py.
    hop_length = 256
    frames_per_second = sample_rate / hop_length

    feature_frames_per_window = int(
        round(window_seconds * frames_per_second)
    )

    total_feature_frames = features.shape[1]
    output_dir.mkdir(parents=True, exist_ok=True)

    window_index = 0
    saved_windows = 0

    for start in range(
        0, total_feature_frames, feature_frames_per_window
    ):
        end = min(
            start + feature_frames_per_window,
            total_feature_frames,
        )

        segment = features[:, start:end]

        if segment.shape[1] == 0:
            continue

        output_path = output_dir / f"window_{window_index:04d}.npy"
        np.save(output_path, segment)

        start_time = start / frames_per_second
        end_time = end / frames_per_second

        print(
            f"Window {window_index}: "
            f"{start_time:.2f}s - {end_time:.2f}s, "
            f"shape={segment.shape}"
        )

        window_index += 1
        saved_windows += 1

    print("\nAudio segmentation completed.")
    print(f"Total windows saved: {saved_windows}")
    print(f"Output directory: {output_dir.resolve()}")

    return saved_windows


if __name__ == "__main__":
    segment_audio_features(
        PROJECT_ROOT
        / "data"
        / "processed"
        / "audio"
        / "In the name of love_logmel.npy",
        PROJECT_ROOT
        / "data"
        / "processed"
        / "audio_windows",
    )
