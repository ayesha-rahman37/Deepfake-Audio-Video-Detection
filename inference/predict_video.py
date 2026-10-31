from pathlib import Path
import subprocess
import tempfile
import warnings

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

THRESHOLD = 0.5


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

def extract_window(
    video_path,
    start,
    end,
    settings,
    transform,
    ffmpeg,
):

    cap = cv2.VideoCapture(str(video_path))

    try:

        if not cap.isOpened():
            raise RuntimeError(
                "Could not open uploaded video."
            )

        fps = cap.get(cv2.CAP_PROP_FPS)

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if (
            not np.isfinite(fps)
            or fps <= 0
            or total_frames <= 0
        ):
            raise RuntimeError(
                "Video FPS or frame count is invalid."
            )

        duration = total_frames / fps

        start = max(0.0, float(start))
        end = min(float(end), duration)

        if end <= start:
            raise RuntimeError(
                "Invalid video window."
            )

        last_valid_time = max(
            start,
            end - 1.0 / fps,
        )

        timestamps = np.linspace(
            start,
            last_valid_time,
            int(settings["frames_per_window"]),
        )

        frames = []

        for timestamp in timestamps:

            frame_index = min(
                max(int(timestamp * fps), 0),
                total_frames - 1,
            )

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_index,
            )

            success, frame = cap.read()

            if not success or frame is None:
                raise RuntimeError(
                    f"Could not decode video frame "
                    f"at {timestamp:.2f}s."
                )

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            image = Image.fromarray(rgb)

            frames.append(
                transform(image)
            )

        video_tensor = torch.stack(frames)

    finally:
        cap.release()

    # ----------------------------------------------
    # AUDIO EXTRACTION
    # ----------------------------------------------

    sample_rate = int(
        settings["audio_sample_rate"]
    )

    target_samples = round(
        float(settings["window_duration"]) * sample_rate
    )

    with tempfile.TemporaryDirectory() as temp_dir:

        wav_path = Path(temp_dir) / "audio.wav"

        command = [
            ffmpeg,
            "-y",
            "-v", "error",
            "-ss", f"{start:.6f}",
            "-i", str(video_path),
            "-t", f"{end - start:.6f}",
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
                "Audio extraction failed. "
                "Check that the video contains a readable audio stream. "
                + (result.stderr or "")[-500:]
            )

        audio, _ = librosa.load(
            str(wav_path),
            sr=sample_rate,
            mono=True,
        )

    if audio.size == 0:
        raise RuntimeError(
            "The extracted audio is empty."
        )

    # Maintain fixed-length audio input for every window.

    if len(audio) < target_samples:

        audio = np.pad(
            audio,
            (0, target_samples - len(audio)),
        )

    else:

        audio = audio[:target_samples]

    # ----------------------------------------------
    # LOG-MEL SPECTROGRAM
    # ----------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
        power=2.0,
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max,
    ).astype(np.float32)

    if not np.isfinite(log_mel).all():
        raise RuntimeError(
            "Audio features contain invalid values."
        )

    return (
        video_tensor,
        torch.from_numpy(log_mel),
    )


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

        duration = frame_count / fps

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

        window_prediction = (
            "FAKE"
            if probability_fake >= THRESHOLD
            else "REAL"
        )

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

    final_prediction = (
        "FAKE"
        if final_probability >= THRESHOLD
        else "REAL"
    )

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

        "warning": (
            "Research prototype: scores are model outputs, "
            "not calibrated probabilities or proof of authenticity."
        ),

    }