
from pathlib import Path
import subprocess

import soundfile as sf
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def extract_audio(video_path, output_path):
    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    config = load_config()
    sample_rate = config["data"]["audio_sample_rate"]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use FFmpeg bundled with imageio-ffmpeg.
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-y",
        "-i", str(video_path),
        "-map", "0:a:0",
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=600,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Audio extraction failed. The video may not contain "
            "an audio stream.\n" + result.stderr[-2000:]
        )

    audio, actual_sample_rate = sf.read(
        str(output_path),
        dtype="float32",
    )

    duration = len(audio) / actual_sample_rate

    print("\nAudio extraction successful!")
    print(f"Output file: {output_path}")
    print(f"Sample rate: {actual_sample_rate} Hz")
    print(f"Channels: 1 (mono)")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Audio samples: {len(audio):,}")
    print(f"File size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")

    return output_path


if __name__ == "__main__":
    extract_audio(
        PROJECT_ROOT / "data" / "raw" / "In the name of love.mp4",
        PROJECT_ROOT / "data" / "processed" / "audio" / "In the name of love.wav",
    )
