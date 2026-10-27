
from pathlib import Path
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
import librosa
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from models.multimodal_model import MultimodalDeepfakeDetector


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
CHECKPOINT_PATH = ROOT / "checkpoints" / "multimodal_reliability_best.pt"


def load_settings():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)["data"]


def build_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_model(device):
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = MultimodalDeepfakeDetector(
        pretrained_visual=False,
        freeze_visual=True,
        frame_batch_size=4,
        fusion_mode="reliability",
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def extract_window(video_path, start, end, settings, transform, ffmpeg):
    cap = cv2.VideoCapture(str(video_path))

    try:
        if not cap.isOpened():
            raise RuntimeError("Could not open uploaded video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0:
            raise RuntimeError("Video FPS or frame count is invalid.")

        duration = total_frames / fps
        start = max(0.0, start)
        end = min(end, duration)

        if end <= start:
            raise RuntimeError("Invalid video window.")

        last_valid_time = max(start, end - 1.0 / fps)
        timestamps = np.linspace(
            start, last_valid_time,
            int(settings["frames_per_window"]),
        )

        frames = []
        last_image = None

        for timestamp in timestamps:
            frame_index = min(
                max(int(timestamp * fps), 0),
                total_frames - 1,
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = cap.read()

            if success and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                last_image = Image.fromarray(rgb)
            elif last_image is None:
                raise RuntimeError(
                    f"Could not decode frame at {timestamp:.2f}s."
                )

            frames.append(transform(last_image))

        video_tensor = torch.stack(frames)
    finally:
        cap.release()

    sample_rate = int(settings["audio_sample_rate"])
    target_samples = round(
        float(settings["window_duration"]) * sample_rate
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = Path(temp_dir) / "audio.wav"

        command = [
            ffmpeg, "-y", "-v", "error",
            "-ss", f"{start:.6f}",
            "-i", str(video_path),
            "-t", f"{end - start:.6f}",
            "-map", "0:a:0",
            "-vn", "-ac", "1",
            "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            str(wav_path),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )

        if result.returncode != 0 or not wav_path.is_file():
            raise RuntimeError(
                "Audio extraction failed. Check that the video "
                "contains a readable audio stream. "
                + (result.stderr or "")[-500:]
            )

        audio, _ = librosa.load(
            str(wav_path),
            sr=sample_rate,
            mono=True,
        )

    if audio.size == 0:
        raise RuntimeError("The extracted audio is empty.")

    if len(audio) < target_samples:
        audio = np.pad(
            audio, (0, target_samples - len(audio))
        )
    else:
        audio = audio[:target_samples]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
        power=2.0,
    )

    log_mel = librosa.power_to_db(
        mel, ref=np.max
    ).astype(np.float32)

    if not np.isfinite(log_mel).all():
        raise RuntimeError("Audio features contain invalid values.")

    return video_tensor, torch.from_numpy(log_mel)


def predict_video(video_path):
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    settings = load_settings()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = load_model(device)
    transform = build_transform(int(settings["image_size"]))
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError("Could not open video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            raise RuntimeError("Invalid video properties.")

        duration = frame_count / fps
    finally:
        cap.release()

    window_seconds = float(settings["window_duration"])
    results = []

    for window_index, start in enumerate(
        np.arange(0, duration, window_seconds)
    ):
        end = min(start + window_seconds, duration)

        video, audio = extract_window(
            video_path, float(start), float(end),
            settings, transform, ffmpeg,
        )

        with torch.inference_mode():
            output = model(
                video.unsqueeze(0).to(device),
                audio.unsqueeze(0).to(device),
            )

            probability_fake = torch.sigmoid(
                output["logits"]
            ).item()

            weights = output["reliability"][0].detach().cpu().tolist()

        results.append({
            "window": window_index + 1,
            "start_seconds": round(float(start), 2),
            "end_seconds": round(float(end), 2),
            "fake_score": round(probability_fake * 100, 2),
            "prediction": (
                "FAKE" if probability_fake >= 0.5 else "REAL"
            ),
            "video_reliability": round(float(weights[0]), 4),
            "audio_reliability": round(float(weights[1]), 4),
        })

    if not results:
        raise RuntimeError("No video windows were processed.")

    average_fake_score = float(np.mean([
        item["fake_score"] for item in results
    ]))

    return {
        "prediction": (
            "FAKE" if average_fake_score >= 50 else "REAL"
        ),
        "fake_score": round(average_fake_score, 2),
        "real_score": round(100 - average_fake_score, 2),
        "duration_seconds": round(duration, 2),
        "device": str(device),
        "windows": results,
        "warning": (
            "Research prototype: scores are model outputs, "
            "not calibrated probabilities or proof of authenticity."
        ),
    }
