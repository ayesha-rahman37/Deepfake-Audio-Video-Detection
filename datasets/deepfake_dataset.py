
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DeepfakeDataset(Dataset):
    """
    Loads matching video-frame windows and audio-feature windows.

    Expected structure:
        data/processed/frames/window_0000/frame_00.jpg
        data/processed/audio_windows/window_0000.npy

    Labels are optional for preprocessing tests.
    Training requires verified real/fake labels.
    """

    def __init__(
        self,
        video_frames_dir=None,
        audio_windows_dir=None,
        labels=None,
        transform=None,
        frames_per_window=16,
    ):
        self.video_frames_dir = Path(
            video_frames_dir
            or PROJECT_ROOT / "data" / "processed" / "frames"
        )

        self.audio_windows_dir = Path(
            audio_windows_dir
            or PROJECT_ROOT / "data" / "processed" / "audio_windows"
        )

        self.labels = labels or {}
        self.transform = transform
        self.frames_per_window = frames_per_window

        if not self.video_frames_dir.exists():
            raise FileNotFoundError(
                f"Video frames directory not found: "
                f"{self.video_frames_dir}"
            )

        if not self.audio_windows_dir.exists():
            raise FileNotFoundError(
                f"Audio windows directory not found: "
                f"{self.audio_windows_dir}"
            )

        self.samples = []

        video_windows = sorted(
            path for path in self.video_frames_dir.glob("window_*")
            if path.is_dir()
        )

        for video_window in video_windows:
            window_name = video_window.name
            audio_path = (
                self.audio_windows_dir / f"{window_name}.npy"
            )

            if not audio_path.exists():
                continue

            frame_paths = sorted(video_window.glob("frame_*.jpg"))

            if not frame_paths:
                continue

            self.samples.append(
                {
                    "window_name": window_name,
                    "frame_paths": frame_paths,
                    "audio_path": audio_path,
                }
            )

        if not self.samples:
            raise RuntimeError(
                "No matching video/audio windows found. "
                "Check the preprocessing output folders."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        frames = []

        for frame_path in sample["frame_paths"]:
            with Image.open(frame_path) as image:
                image = image.convert("RGB")

                if self.transform is not None:
                    frame = self.transform(image)
                else:
                    # Convert RGB image to float tensor in [0, 1].
                    image_array = np.asarray(image).copy()
                    frame = torch.from_numpy(image_array)
                    frame = frame.permute(2, 0, 1).float() / 255.0

                frames.append(frame)

        video_tensor = torch.stack(frames)

        audio_array = np.load(sample["audio_path"]).astype(
            np.float32
        )
        audio_tensor = torch.from_numpy(audio_array)

        item = {
            "window_name": sample["window_name"],
            "video": video_tensor,
            "audio": audio_tensor,
        }

        if sample["window_name"] in self.labels:
            item["label"] = torch.tensor(
                self.labels[sample["window_name"]],
                dtype=torch.long,
            )

        return item


if __name__ == "__main__":
    dataset = DeepfakeDataset()

    print("Dataset loading test successful!")
    print(f"Matching windows: {len(dataset)}")

    sample = dataset[0]

    print(f"First window: {sample['window_name']}")
    print(f"Video tensor shape: {tuple(sample['video'].shape)}")
    print(f"Audio tensor shape: {tuple(sample['audio'].shape)}")
    print(f"Label available: {'label' in sample}")
