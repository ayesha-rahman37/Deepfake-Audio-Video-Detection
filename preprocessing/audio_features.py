
from pathlib import Path

import librosa
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def extract_log_mel(audio_path, output_path):
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    config = load_config()
    sample_rate = config["data"]["audio_sample_rate"]

    # Load mono audio at the configured sample rate.
    audio, sr = librosa.load(
        str(audio_path),
        sr=sample_rate,
        mono=True,
    )

    if audio.size == 0:
        raise ValueError("The audio file is empty.")

    # Calculate the log-Mel spectrogram.
    mel_power = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
        power=2.0,
    )

    log_mel = librosa.power_to_db(
        mel_power,
        ref=np.max,
    ).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, log_mel)

    print("\nLog-Mel feature extraction successful!")
    print(f"Input audio: {audio_path}")
    print(f"Sample rate: {sr} Hz")
    print(f"Audio duration: {len(audio) / sr:.2f} seconds")
    print(f"Feature shape: {log_mel.shape}")
    print(f"Feature dtype: {log_mel.dtype}")
    print(f"Saved to: {output_path.resolve()}")
    print(f"Feature file size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")

    return log_mel


if __name__ == "__main__":
    extract_log_mel(
        PROJECT_ROOT / "data" / "processed" / "audio" / "In the name of love.wav",
        PROJECT_ROOT / "data" / "processed" / "audio" / "In the name of love_logmel.npy",
    )
