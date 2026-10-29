"""PyTorch inference for the multimodal deepfake detector.

A trained checkpoint is required before any prediction can be returned.
"""

from pathlib import Path
import tempfile

import cv2
import librosa
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from models.multimodal_model import MultimodalDeepfakeDetector
from preprocessing.frame_sampler import sample_frames
from preprocessing.audio_extractor import extract_audio
from preprocessing.audio_features import extract_log_mel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "multimodal_reliability_best.pt"
)


def _load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _load_detector(checkpoint_path):
    """Load a trained model. Never substitute random weights."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Trained model checkpoint not found: {checkpoint_path}. "
            "Train the model and place its checkpoint at this path."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint format.")

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint has no model_state_dict.")

    fusion_mode = checkpoint.get("fusion_mode", "reliability")
    if fusion_mode not in {"concat", "reliability"}:
        raise ValueError("Checkpoint contains an unsupported fusion mode.")

    model = MultimodalDeepfakeDetector(
        pretrained_visual=False,
        freeze_visual=True,
        frame_batch_size=4,
        fusion_mode=fusion_mode,
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, device


def _video_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def _load_window_frames(window_dir, image_size, expected_frames):
    """Read sampled frames and return [1, T, 3, H, W]."""
    frame_paths = sorted(window_dir.glob("frame_*.jpg"))

    if not frame_paths:
        raise RuntimeError(
            f"No readable sampled frames in {window_dir.name}."
        )

    transform = _video_transform(image_size)
    frames = []

    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path))
        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        frames.append(transform(image))

    if not frames:
        raise RuntimeError(
            f"Could not read any sampled frames in {window_dir.name}."
        )

    # Keep a consistent frame count if some video frames could not be read.
    if len(frames) >= expected_frames:
        indices = np.linspace(
            0, len(frames) - 1, expected_frames, dtype=int
        )
        frames = [frames[index] for index in indices]
    else:
        frames.extend([frames[-1]] * (expected_frames - len(frames)))

    return torch.stack(frames).unsqueeze(0)


def _audio_window(log_mel, window_index, window_seconds,
                  sample_rate, hop_length, expected_audio_frames):
    """Extract one time-aligned [1, 64, F] log-Mel window."""
    start = int(
        window_index * window_seconds * sample_rate / hop_length
    )
    end = start + expected_audio_frames
    segment = log_mel[:, start:end]

    if segment.shape[1] == 0:
        raise RuntimeError(
            f"No audio features available for window {window_index}."
        )

    if segment.shape[1] < expected_audio_frames:
        segment = np.pad(
            segment,
            ((0, 0), (0, expected_audio_frames - segment.shape[1])),
            mode="constant",
            constant_values=-80.0,
        )

    segment = np.ascontiguousarray(segment, dtype=np.float32)
    return torch.from_numpy(segment).unsqueeze(0)


def analyze_video(video_path, checkpoint_path=None):
    """Analyze a video and return label, confidence, and window evidence.

    Raises an exception if the video cannot be processed or the trained
    checkpoint is unavailable.
    """
    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT

    # Check the checkpoint before doing expensive preprocessing.
    model, device = _load_detector(checkpoint_path)

    config = _load_config()
    data_config = config["data"]

    window_seconds = float(data_config["window_duration"])
    frames_per_window = int(data_config["frames_per_window"])
    image_size = int(data_config["image_size"])
    sample_rate = int(data_config["audio_sample_rate"])
    hop_length = 256

    expected_audio_frames = int(
        round(window_seconds * sample_rate / hop_length)
    ) + 1

    with tempfile.TemporaryDirectory(prefix="deepfake_inference_") as temp:
        temp_dir = Path(temp)
        frames_dir = temp_dir / "frames"
        audio_path = temp_dir / "audio.wav"
        mel_path = temp_dir / "log_mel.npy"

        sample_frames(video_path, frames_dir)
        extract_audio(video_path, audio_path)
        log_mel = extract_log_mel(audio_path, mel_path)

        if not isinstance(log_mel, np.ndarray) or log_mel.ndim != 2:
            raise RuntimeError("Audio preprocessing returned invalid features.")

        if log_mel.shape[0] != 64:
            raise RuntimeError(
                f"Expected 64 Mel bands, got {log_mel.shape[0]}."
            )

        window_dirs = sorted(frames_dir.glob("window_*"))
        if not window_dirs:
            raise RuntimeError("No video windows were generated.")

        evidence = []
        fake_probabilities = []

        with torch.inference_mode():
            for window_dir in window_dirs:
                try:
                    window_index = int(window_dir.name.split("_")[-1])
                except ValueError as exc:
                    raise RuntimeError(
                        f"Invalid window directory: {window_dir.name}"
                    ) from exc

                video_tensor = _load_window_frames(
                    window_dir, image_size, frames_per_window
                ).to(device)

                audio_tensor = _audio_window(
                    log_mel,
                    window_index,
                    window_seconds,
                    sample_rate,
                    hop_length,
                    expected_audio_frames,
                ).to(device)

                output = model(video_tensor, audio_tensor)
                logits = output["logits"].reshape(-1)

                if logits.numel() != 1 or not torch.isfinite(logits).all():
                    raise RuntimeError(
                        f"Invalid model output for window {window_index}."
                    )

                # Training labels: 0 = Real, 1 = Fake.
                fake_probability = torch.sigmoid(logits)[0].item()
                label = "Fake" if fake_probability >= 0.5 else "Real"
                confidence = (
                    fake_probability
                    if label == "Fake"
                    else 1.0 - fake_probability
                )

                fake_probabilities.append(fake_probability)

                reliability = output.get("reliability")
                reliability_values = None
                if reliability is not None:
                    reliability_values = (
                        reliability[0].detach().cpu().tolist()
                    )

                evidence.append({
                    "window": window_index + 1,
                    "label": label,
                    "confidence": float(confidence),
                    "fake_probability": float(fake_probability),
                    "reliability": reliability_values,
                })

        # Aggregate window scores with equal weighting.
        overall_fake_probability = float(np.mean(fake_probabilities))
        overall_label = (
            "Fake" if overall_fake_probability >= 0.5 else "Real"
        )
        overall_confidence = (
            overall_fake_probability
            if overall_label == "Fake"
            else 1.0 - overall_fake_probability
        )

        return {
            "label": overall_label,
            "confidence": overall_confidence,
            "window_evidence": evidence,
            "device": str(device),
        }