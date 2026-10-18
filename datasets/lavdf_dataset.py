import os

import csv
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import librosa
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(
    os.environ.get("LAVDF_ROOT", r"D:\LAV-DF")
).expanduser()
METADATA_CSV = PROJECT_ROOT / "data" / "metadata" / "lavdf_metadata.csv"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class LAVDFDataset(Dataset):
    """
    Loads aligned full-frame video and audio windows from LAV-DF.

    Each item represents a 5-second window (or a shorter final window).
    A window is labelled fake when it overlaps an annotated fake period.
    """

    def __init__(self, split="train"):
        if split not in {"train", "dev", "test"}:
            raise ValueError("split must be train, dev, or test")

        if not METADATA_CSV.is_file():
            raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")

        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        settings = config["data"]
        self.window_seconds = float(settings["window_duration"])
        self.frames_per_window = int(settings["frames_per_window"])
        self.image_size = int(settings["image_size"])
        self.sample_rate = int(settings["audio_sample_rate"])

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        self.samples = []

        with METADATA_CSV.open("r", encoding="utf-8", newline="") as file:
            records = csv.DictReader(file)

            for record in records:
                if record["split"] != split:
                    continue

                duration = float(record["duration"])
                if duration <= 0:
                    continue

                video_path = DATASET_ROOT / record["video_path"]
                if not video_path.is_file():
                    continue

                fake_periods = json.loads(record["fake_periods"])

                start = 0.0
                while start < duration:
                    end = min(start + self.window_seconds, duration)

                    # Positive if the window overlaps any fake segment.
                    is_fake = any(
                        float(period[0]) < end
                        and float(period[1]) > start
                        for period in fake_periods
                    )

                    self.samples.append({
                        "video_path": video_path,
                        "sample_id": record["sample_id"],
                        "start": start,
                        "end": end,
                        "label": int(is_fake),
                    })

                    start = end

        if not self.samples:
            raise RuntimeError(f"No usable samples found for split={split}")

        print(
            f"LAV-DF {split}: {len(self.samples)} windows "
            f"from local videos"
        )

    def __len__(self):
        return len(self.samples)

    def _load_video(self, video_path, start, end):
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or frame_count <= 0:
            cap.release()
            raise RuntimeError(f"Invalid video properties: {video_path}")

        actual_duration = frame_count / fps
        clip_end = min(end, actual_duration)

        if clip_end <= start:
            cap.release()
            raise RuntimeError(f"Empty video window: {video_path}")

        # Uniformly sample full frames across this time window.
        times = np.linspace(
            start,
            max(start, clip_end - 1.0 / fps),
            self.frames_per_window,
        )

        frames = []
        last_frame = None

        for timestamp in times:
            frame_index = min(
                int(timestamp * fps),
                frame_count - 1,
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = cap.read()

            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                last_frame = Image.fromarray(frame)
            elif last_frame is not None:
                frame = last_frame.copy()
            else:
                cap.release()
                raise RuntimeError(
                    f"Cannot decode frame at {timestamp:.3f}s: {video_path}"
                )

            if not success:
                image = frame
            else:
                image = last_frame

            frames.append(self.transform(image))

        cap.release()
        return torch.stack(frames)

    def _load_audio(self, video_path, start, end):
        # Extract only the matching audio segment to a temporary WAV file.
        segment_duration = end - start

        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "segment.wav"

            command = [
                self.ffmpeg,
                "-y",
                "-v", "error",
                "-ss", str(start),
                "-i", str(video_path),
                "-t", str(segment_duration),
                "-vn",
                "-ac", "1",
                "-ar", str(self.sample_rate),
                "-c:a", "pcm_s16le",
                str(wav_path),
            ]

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0 or not wav_path.is_file():
                raise RuntimeError(
                    f"Audio extraction failed for {video_path}: "
                    f"{result.stderr[-500:]}"
                )

            audio, _ = librosa.load(
                str(wav_path),
                sr=self.sample_rate,
                mono=True,
            )

        target_samples = round(self.window_seconds * self.sample_rate)

        # Pad short final windows so the model receives a fixed input shape.
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))
        else:
            audio = audio[:target_samples]

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=64,
            power=2.0,
        )

        log_mel = librosa.power_to_db(mel, ref=np.max)
        return torch.from_numpy(log_mel.astype(np.float32))

    def __getitem__(self, index):
        sample = self.samples[index]

        video = self._load_video(
            sample["video_path"],
            sample["start"],
            sample["end"],
        )

        audio = self._load_audio(
            sample["video_path"],
            sample["start"],
            sample["end"],
        )

        return {
            "sample_id": sample["sample_id"],
            "start_seconds": sample["start"],
            "end_seconds": sample["end"],
            "video": video,              # (16, 3, 224, 224)
            "audio": audio,              # (64, time_frames)
            "label": torch.tensor(sample["label"], dtype=torch.long),
        }


if __name__ == "__main__":
    dataset = LAVDFDataset(split="train")
    sample = dataset[0]

    print("\nLAV-DF sample test successful!")
    print("Video:", tuple(sample["video"].shape))
    print("Audio:", tuple(sample["audio"].shape))
    print("Label:", sample["label"].item())
    print("Window:", sample["start_seconds"], "-", sample["end_seconds"])
