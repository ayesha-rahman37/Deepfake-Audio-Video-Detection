"""Run window-level multimodal deepfake inference on a video file.

Example:
    python inference/predict_video.py --video path/to/video.mp4
    python inference/predict_video.py --video path/to/video.mp4 --output results/prediction.json
"""

import argparse
import json
import sys
import subprocess
import tempfile
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import imageio_ffmpeg
import librosa
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from models.multimodal_model import MultimodalDeepfakeDetector

ROOT = PROJECT_ROOT
CONFIG_PATH = ROOT / "configs" / "config.yaml"
CHECKPOINT_PATH = ROOT / "checkpoints" / "multimodal_reliability_best.pt"
THRESHOLD = 0.5
UNCERTAINTY_MARGIN = 0.10

def classify_score(probability_fake):
    """Return a cautious label; scores near 0.5 remain inconclusive."""
    if probability_fake >= THRESHOLD + UNCERTAINTY_MARGIN:
        return "FAKE"
    if probability_fake <= THRESHOLD - UNCERTAINTY_MARGIN:
        return "REAL"
    return "INCONCLUSIVE"


# --------------------------------------------------
# 1. LOAD CONFIGURATION
# --------------------------------------------------

def load_settings():

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or "data" not in config:
        raise ValueError(
            "Invalid configuration: 'data' section is missing."
        )

    settings = config["data"]

    required = [
        "window_duration",
        "frames_per_window",
        "image_size",
        "audio_sample_rate",
    ]

    for key in required:
        if key not in settings:
            raise ValueError(
                f"Missing configuration setting: {key}"
            )

    if float(settings["window_duration"]) <= 0:
        raise ValueError("window_duration must be positive.")

    if int(settings["frames_per_window"]) < 1:
        raise ValueError("frames_per_window must be positive.")

    if int(settings["image_size"]) < 1:
        raise ValueError("image_size must be positive.")

    if int(settings["audio_sample_rate"]) < 1:
        raise ValueError("audio_sample_rate must be positive.")

    return settings


# --------------------------------------------------
# 2. BUILD IMAGE TRANSFORM
# --------------------------------------------------

def build_transform(image_size):

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# --------------------------------------------------
# 3. LOAD TRAINED MODEL
# --------------------------------------------------

def load_model(device):

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # The checkpoint contains the trained model weights.
    # Pretrained initialization is not needed before loading
    # a complete trained state dictionary.

    model = MultimodalDeepfakeDetector(
        pretrained_visual=False,
        freeze_visual=True,
        frame_batch_size=4,
        fusion_mode="reliability",
    ).to(device)

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
        )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Invalid checkpoint format."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain model_state_dict."
        )

    saved_fusion_mode = checkpoint.get("fusion_mode")

    if saved_fusion_mode is not None:
        if saved_fusion_mode != "reliability":
            raise RuntimeError(
                "Checkpoint fusion mode mismatch. "
                f"Expected reliability, got {saved_fusion_mode}."
            )
    else:
        warnings.warn(
            "Checkpoint does not contain fusion_mode metadata. "
            "Model compatibility is being checked through its weights."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    return model


# --------------------------------------------------
# 4. EXTRACT VIDEO FRAMES AND AUDIO FEATURES
# --------------------------------------------------

def _read_frame_with_ffmpeg(video_path, timestamp, ffmpeg, max_retries=8, step_seconds=0.12):
    """Decode a frame with FFmpeg, retrying earlier timestamps near EOF."""
    video_path = str(video_path)

    for attempt in range(max_retries + 1):
        seek_time = max(0.0, float(timestamp) - attempt * step_seconds)

        with tempfile.TemporaryDirectory() as temp_dir:
            frame_path = Path(temp_dir) / "frame.jpg"

            # Place -ss after -i for accurate seeking. This is slower than
            # input seeking but more reliable for short/mobile MP4 recordings.
            command = [
                ffmpeg,
                "-y",
                "-v", "error",
                "-i", video_path,
                "-ss", f"{seek_time:.6f}",
                "-frames:v", "1",
                "-q:v", "2",
                str(frame_path),
            ]

            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                continue

            if result.returncode == 0 and frame_path.is_file():
                frame = cv2.imread(str(frame_path))
                if frame is not None and frame.size > 0:
                    return frame, seek_time

    return None, None


def extract_window(
    video_path,
    start,
    end,
    settings,
    transform,
    ffmpeg,
):
    """Extract a fixed number of valid frames and aligned audio for one window."""
    cap = cv2.VideoCapture(str(video_path))

    try:
        if not cap.isOpened():
            raise RuntimeError("Could not open uploaded video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not np.isfinite(fps) or fps <= 0 or total_frames < 2:
            raise RuntimeError("Video FPS or frame count is invalid.")

        # Use the last frame before the container's reported end. This avoids
        # seeking to an MP4 timestamp that exists in metadata but has no frame.
        last_decodable_time = max(0.0, (total_frames - 2) / fps)
        start = max(0.0, float(start))
        end = min(float(end), last_decodable_time + 1.0 / fps)

        if end <= start:
            raise RuntimeError("Invalid video window after duration adjustment.")

        last_sample_time = min(last_decodable_time, max(start, end - 1.0 / fps))
        frame_count = int(settings["frames_per_window"])
        timestamps = np.linspace(start, last_sample_time, frame_count)
        frames = []

        for timestamp in timestamps:
            # Clamp every request to a frame index that should exist.
            frame_index = int(round(float(timestamp) * fps))
            frame_index = min(max(frame_index, 0), total_frames - 2)
            frame = None

            # First try OpenCV. Retry nearby earlier frames because some
            # variable-frame-rate/mobile MP4s fail on direct end-of-file seeks.
            for offset in range(0, min(31, total_frames - 1)):
                candidate_index = max(0, frame_index - offset)
                cap.set(cv2.CAP_PROP_POS_FRAMES, candidate_index)
                ok, candidate = cap.read()
                if ok and candidate is not None and candidate.size > 0:
                    frame = candidate
                    break

            # FFmpeg fallback: accurate seeking and multiple earlier retries.
            if frame is None:
                frame, _ = _read_frame_with_ffmpeg(
                    video_path,
                    min(float(timestamp), last_decodable_time),
                    ffmpeg,
                )

            if frame is None:
                raise RuntimeError(
                    f"Could not decode a video frame near {timestamp:.2f}s "
                    "even after OpenCV and FFmpeg retries. "
                    "The uploaded file may be damaged or use an unsupported codec."
                )

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(Image.fromarray(rgb)))

        video_tensor = torch.stack(frames)

    finally:
        cap.release()

    # AUDIO EXTRACTION: retain the original window start/end for A/V alignment.
    sample_rate = int(settings["audio_sample_rate"])
    target_samples = round(float(settings["window_duration"]) * sample_rate)

    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = Path(temp_dir) / "audio.wav"
        command = [
            ffmpeg,
            "-y",
            "-v", "error",
            "-ss", f"{start:.6f}",
            "-i", str(video_path),
            "-t", f"{max(0.01, end - start):.6f}",
            "-map", "0:a:0",
            "-vn",
            "-ac", "1",
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
                "Audio extraction failed. Check that the video contains a "
                "readable audio stream. " + (result.stderr or "")[-500:]
            )

        audio, _ = librosa.load(str(wav_path), sr=sample_rate, mono=True)

    if audio.size == 0:
        raise RuntimeError("The extracted audio is empty.")

    # Fixed-length audio tensor expected by the trained model.
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
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
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    if not np.isfinite(log_mel).all():
        raise RuntimeError("Audio features contain invalid values.")

    return video_tensor, torch.from_numpy(log_mel)


# --------------------------------------------------
# 5. MAIN VIDEO PREDICTION
# --------------------------------------------------

def predict_video(video_path):

    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    settings = load_settings()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(device)

    transform = build_transform(
        int(settings["image_size"])
    )

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    # ----------------------------------------------
    # VIDEO METADATA
    # ----------------------------------------------

    cap = cv2.VideoCapture(str(video_path))

    try:

        if not cap.isOpened():
            raise RuntimeError(
                "Could not open video."
            )

        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if (
            not np.isfinite(fps)
            or fps <= 0
            or frame_count <= 0
        ):
            raise RuntimeError(
                "Invalid video properties."
            )

        # OpenCV's reported frame count can include an undecodable trailing
        # frame. Use a conservative duration for window boundaries.
        duration = max(0.0, (frame_count - 2) / fps)

        if not np.isfinite(duration) or duration <= 0:
            raise RuntimeError(
                "Invalid video duration."
            )

    finally:
        cap.release()

    # ----------------------------------------------
    # PROCESS VIDEO WINDOWS
    # ----------------------------------------------

    window_seconds = float(
        settings["window_duration"]
    )

    results = []

    window_starts = np.arange(
        0,
        duration,
        window_seconds,
    )

    for window_index, start in enumerate(window_starts):

        start = float(start)

        end = min(
            start + window_seconds,
            duration,
        )

        video, audio = extract_window(
            video_path,
            start,
            end,
            settings,
            transform,
            ffmpeg,
        )

        # ------------------------------------------
        # MODEL INFERENCE
        # ------------------------------------------

        with torch.inference_mode():

            output = model(
                video.unsqueeze(0).to(device),
                audio.unsqueeze(0).to(device),
            )

            probability_fake = torch.sigmoid(
                output["logits"]
            ).item()

            reliability = (
                output["reliability"][0]
                .detach()
                .cpu()
                .tolist()
            )

        if not np.isfinite(probability_fake):
            raise RuntimeError(
                f"Invalid prediction in window {window_index + 1}."
            )

        if len(reliability) != 2:
            raise RuntimeError(
                "Invalid modality reliability output."
            )

        video_reliability = float(reliability[0])
        audio_reliability = float(reliability[1])

        if not np.isfinite(reliability).all():
            raise RuntimeError(
                "Model returned invalid reliability values."
            )

        fake_score = probability_fake * 100

        window_prediction = classify_score(probability_fake)

        results.append({

            "window": window_index + 1,

            "start_seconds": round(start, 2),

            "end_seconds": round(end, 2),

            "fake_score": round(fake_score, 2),

            "prediction": window_prediction,

            "video_reliability": round(
                video_reliability,
                4,
            ),

            "audio_reliability": round(
                audio_reliability,
                4,
            ),

        })

    if not results:
        raise RuntimeError(
            "No video windows were processed."
        )

    # ----------------------------------------------
    # DURATION-WEIGHTED FINAL PREDICTION
    # ----------------------------------------------

    window_durations = np.array(
        [
            item["end_seconds"] - item["start_seconds"]
            for item in results
        ],
        dtype=np.float64,
    )

    window_scores = np.array(
        [
            item["fake_score"]
            for item in results
        ],
        dtype=np.float64,
    )

    if (
        not np.isfinite(window_durations).all()
        or not np.isfinite(window_scores).all()
        or np.any(window_durations <= 0)
    ):
        raise RuntimeError(
            "Invalid window scores or durations."
        )

    average_fake_score = float(
        np.average(
            window_scores,
            weights=window_durations,
        )
    )

    final_probability = average_fake_score / 100.0

    final_prediction = classify_score(final_probability)

    # ----------------------------------------------
    # FINAL RESULT
    # ----------------------------------------------

    return {

        "prediction": final_prediction,

        "fake_score": round(
            average_fake_score,
            2,
        ),

        "real_score": round(
            100 - average_fake_score,
            2,
        ),

        "duration_seconds": round(
            duration,
            2,
        ),

        "device": str(device),

        "windows": results,

        "decision_thresholds": {
            "likely_real_max_fake_score": round((THRESHOLD - UNCERTAINTY_MARGIN) * 100, 2),
            "inconclusive_range": [
                round((THRESHOLD - UNCERTAINTY_MARGIN) * 100, 2),
                round((THRESHOLD + UNCERTAINTY_MARGIN) * 100, 2),
            ],
            "likely_fake_min_fake_score": round((THRESHOLD + UNCERTAINTY_MARGIN) * 100, 2),
        },
        "warning": (
            "Research prototype: scores are model outputs, not calibrated "
            "probabilities or proof of authenticity. INCONCLUSIVE means the "
            "score is near the decision boundary. A REAL recording can be "
            "incorrectly flagged as FAKE; verify results independently."
        ),

    }

# --------------------------------------------------
# 6. COMMAND-LINE INTERFACE
# --------------------------------------------------

def main():
    global THRESHOLD
    parser = argparse.ArgumentParser(
        description="Multimodal audio-video deepfake detection"
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to the input video file"
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional JSON output path"
    )
    parser.add_argument(
        "--threshold", type=float, default=THRESHOLD,
        help="Fake decision threshold between 0 and 1 (default: 0.5)"
    )
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    THRESHOLD = args.threshold

    result = predict_video(args.video)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        print(f"\nPrediction saved to: {output_path}")


if __name__ == "__main__":
    main()
