
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

from models.audio_encoder import AudioEncoder
from utils.device import device

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "audio_encoder_candidate.pt"


def normalize(audio):
    return (audio + 40.0) / 40.0


class DummyAudioDataset(Dataset):
    """Random data for pipeline testing only."""

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


def collate_audio_batch(samples):
    # Failed audio samples are None and are excluded from the batch.
    samples = [sample for sample in samples if sample is not None]

    if not samples:
        return None

    audio, labels = zip(*samples)
    return torch.stack(audio), torch.stack(labels)


def build_real_dataset_class():
    from datasets.lavdf_dataset import LAVDFDataset, METADATA_CSV

    class AudioWindowDataset(LAVDFDataset):
        def __init__(self, split="train"):
            super().__init__(split=split)

            self.audio_modified = {}
            with METADATA_CSV.open(
                "r", encoding="utf-8", newline=""
            ) as file:
                for row in csv.DictReader(file):
                    flag = str(row["modify_audio"]).strip().lower()
                    self.audio_modified[row["sample_id"]] = flag in {
                        "1", "true", "yes"
                    }

            self.failed_samples = 0

        def __getitem__(self, index):
            sample = self.samples[index]

            try:
                audio = self._load_audio(
                    sample["video_path"],
                    sample["start"],
                    sample["end"]
                )

                if not isinstance(audio, torch.Tensor):
                    raise RuntimeError("Audio extraction did not return a tensor.")

                if audio.shape != (64, 313):
                    raise RuntimeError(
                        f"Unexpected audio shape: {tuple(audio.shape)}"
                    )

                if not torch.isfinite(audio).all():
                    raise RuntimeError("Audio contains NaN or infinite values.")

            except (RuntimeError, ValueError, OSError) as error:
                self.failed_samples += 1
                if self.failed_samples <= 10:
                    print(
                        f"Skipping {sample['sample_id']}: {error}"
                    )
                return None

            is_fake = (
                sample["label"] == 1
                and self.audio_modified.get(sample["sample_id"], False)
            )

            return audio, torch.tensor(float(is_fake))

    return AudioWindowDataset


def limit(dataset, max_samples, seed=42):
    if max_samples > 0 and len(dataset) > max_samples:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(
            len(dataset), generator=generator
        )[:max_samples]
        return Subset(dataset, indices.tolist())

    return dataset


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    count = 0
    skipped_batches = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            if batch is None:
                skipped_batches += 1
                continue

            audio, label = batch
            audio = normalize(audio).to(device)
            label = label.float().to(device)

            _, logit = model(audio)
            loss = criterion(logit, label)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            predictions = (torch.sigmoid(logit) >= 0.5).float()

            correct += (predictions == label).sum().item()
            count += label.numel()
            total_loss += loss.item() * label.numel()

    if count == 0:
        raise RuntimeError(
            "No valid audio samples were available in this epoch."
        )

    return total_loss / count, correct / count, count, skipped_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum windows per split; 0 means use all."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT)
    )
    args = parser.parse_args()

    if args.max_samples < 0:
        parser.error("--max_samples must be 0 or a positive integer.")

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        settings = yaml.safe_load(file)["training"]

    epochs = args.epochs or int(settings["epochs"])
    batch_size = int(settings["batch_size"])
    lr = float(settings["learning_rate"])

    if epochs < 1:
        parser.error("--epochs must be at least 1.")

    print("Device:", device)

    if args.dummy:
        train_set = DummyAudioDataset(64)
        val_set = DummyAudioDataset(16)
    else:
        dataset_class = build_real_dataset_class()
        train_set = limit(
            dataset_class("train"), args.max_samples
        )
        val_set = limit(
            dataset_class("dev"), args.max_samples
        )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_audio_batch
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_audio_batch
    )

    model = AudioEncoder().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_acc = -1.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_count, train_skipped = run_epoch(
            model, train_loader, criterion, optimizer
        )

        val_loss, val_acc, val_count, val_skipped = run_epoch(
            model, val_loader, criterion
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss {train_loss:.4f} "
            f"acc {train_acc:.3f} samples {train_count} | "
            f"val loss {val_loss:.4f} "
            f"acc {val_acc:.3f} samples {val_count}"
        )

        print(
            f"Skipped batches: train={train_skipped}, "
            f"validation={val_skipped}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print("Best candidate checkpoint saved.")

    print(f"Best validation accuracy: {best_acc:.3f}")
    print(f"Candidate checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
