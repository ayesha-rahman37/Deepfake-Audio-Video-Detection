import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.audio_encoder import AudioEncoder  # noqa: E402
from utils.device import device  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "audio_encoder.pt"


def normalize(audio):
    # Log-Mel values are roughly in [-80, 0] dB; scale to about [-1, 1].
    return (audio + 40.0) / 40.0


class DummyAudioDataset(Dataset):
    """Random data, only to test the training code on CPU without LAV-DF."""

    def __init__(self, size=32):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        label = index % 2
        audio = torch.randn(64, 313) * 10 - 40
        if label == 1:
            audio = audio + 5
        return audio, torch.tensor(float(label))


def build_real_dataset_class():
    # Imported here so --dummy works even without the dataset.
    from datasets.lavdf_dataset import LAVDFDataset, METADATA_CSV

    class AudioWindowDataset(LAVDFDataset):
        """
        Audio-only version of LAVDFDataset.
        Label = 1 only if the window overlaps a fake period AND
        the video's audio was modified (modify_audio).
        """

        def __init__(self, split="train"):
            super().__init__(split=split)
            self.audio_modified = {}
            with METADATA_CSV.open("r", encoding="utf-8", newline="") as file:
                for row in csv.DictReader(file):
                    flag = str(row["modify_audio"]).strip().lower()
                    self.audio_modified[row["sample_id"]] = flag in {
                        "1", "true", "yes"
                    }

        def __getitem__(self, index):
            sample = self.samples[index]
            try:
                audio = self._load_audio(
                    sample["video_path"], sample["start"], sample["end"]
                )
            except RuntimeError as error:
                print(f"Warning: audio failed for {sample['sample_id']}: {error}")
                audio = torch.full((64, 313), -80.0)  # silent placeholder

            is_fake = sample["label"] == 1 and self.audio_modified.get(
                sample["sample_id"], False
            )
            return audio, torch.tensor(float(is_fake))

    return AudioWindowDataset


def limit(dataset, max_samples, seed=42):
    if max_samples and len(dataset) > max_samples:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:max_samples]
        return Subset(dataset, indices.tolist())
    return dataset


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for audio, label in loader:
            audio = normalize(audio).to(device)
            label = label.float().to(device)

            _, logit = model(audio)
            loss = criterion(logit, label)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            predictions = (torch.sigmoid(logit) > 0.5).float()
            correct += (predictions == label).sum().item()
            count += label.numel()
            total_loss += loss.item() * label.numel()

    return total_loss / count, correct / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true",
                        help="use random data (no dataset needed)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=64,
                        help="max training windows (0 = use all)")
    args = parser.parse_args()

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        settings = yaml.safe_load(file)["training"]

    epochs = args.epochs or int(settings["epochs"])
    batch_size = int(settings["batch_size"])
    lr = float(settings["learning_rate"])

    print("Device:", device)

    if args.dummy:
        train_set, val_set = DummyAudioDataset(64), DummyAudioDataset(16)
    else:
        dataset_class = build_real_dataset_class()
        train_set = limit(dataset_class("train"), args.max_samples)
        val_set = limit(dataset_class("dev"), max(args.max_samples // 4, 8))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = AudioEncoder().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, criterion)
        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    print(f"Done. Best val acc: {best_acc:.3f}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()