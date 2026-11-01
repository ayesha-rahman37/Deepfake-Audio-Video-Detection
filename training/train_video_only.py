"""
Video-only LAV-DF baseline: ResNet50 frame encoder + BiLSTM temporal model.

Uses the modality-specific video_label (not the overall audio/video label).
The original multimodal dataset label is preserved separately.
Outputs are versioned so existing candidate checkpoints/results are not overwritten.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.lavdf_dataset import LAVDFDataset
from models.visual_encoder import VisualEncoder
from models.temporal_model import TemporalModel


class VideoOnlyDataset(LAVDFDataset):
    """Load full-frame video and video-specific window labels only."""

    def __getitem__(self, index):
        sample = self.samples[index]
        video = self._load_video(
            sample["video_path"], sample["start"], sample["end"]
        )
        label = int(sample["video_label"])
        return video, torch.tensor(label, dtype=torch.float32)


class VideoOnlyModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.visual = VisualEncoder(
            pretrained=pretrained,
            freeze_backbone=True,
        )
        self.temporal = TemporalModel(input_dim=2048, hidden_dim=256)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, video):
        # Keep the frozen ResNet BatchNorm running statistics fixed even
        # while the temporal head and classifier are in training mode.
        self.visual.eval()

        batch, frames, channels, height, width = video.shape
        flat = video.reshape(batch * frames, channels, height, width)

        features = []
        with torch.no_grad():
            for start in range(0, flat.shape[0], 4):
                features.append(self.visual(flat[start:start + 4]))

        features = torch.cat(features, dim=0)
        features = features.reshape(batch, frames, 2048)
        temporal = self.temporal(features)
        return self.classifier(temporal).squeeze(1)


def select_indices(dataset, limit, seed=42):
    """Select a deterministic, approximately class-balanced subset."""
    labels = [int(s["video_label"]) for s in dataset.samples]
    class_indices = {
        0: [i for i, y in enumerate(labels) if y == 0],
        1: [i for i, y in enumerate(labels) if y == 1],
    }

    if not class_indices[0] or not class_indices[1]:
        raise RuntimeError(
            "Video-only split must contain both classes. "
            f"Counts: real={len(class_indices[0])}, "
            f"video_fake={len(class_indices[1])}. "
            "Check modify_video and fake_periods metadata."
        )

    if limit <= 0 or limit >= len(dataset):
        return list(range(len(dataset)))

    rng = random.Random(seed)
    for values in class_indices.values():
        rng.shuffle(values)

    # Balance the requested subset, while not duplicating samples.
    target_each = limit // 2
    selected = (
        class_indices[0][:target_each] +
        class_indices[1][:target_each]
    )

    if len(selected) < limit:
        selected_set = set(selected)
        remaining = [
            i for i in range(len(dataset)) if i not in selected_set
        ]
        rng.shuffle(remaining)
        selected.extend(remaining[:limit - len(selected)])

    rng.shuffle(selected)
    return selected


def class_counts(dataset, indices):
    labels = [int(dataset.samples[i]["video_label"]) for i in indices]
    return {"real": labels.count(0), "video_fake": labels.count(1)}


def calculate_metrics(labels_all, probs_all, loss):
    predictions = [int(p >= 0.5) for p in probs_all]
    both = len(set(labels_all)) == 2
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(labels_all, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels_all, predictions)
        ),
        "precision_fake": float(
            precision_score(labels_all, predictions, zero_division=0)
        ),
        "recall_fake": float(
            recall_score(labels_all, predictions, zero_division=0)
        ),
        "f1_fake": float(
            f1_score(labels_all, predictions, zero_division=0)
        ),
        "macro_f1": float(
            f1_score(labels_all, predictions, average="macro", zero_division=0)
        ),
        "roc_auc": (
            float(roc_auc_score(labels_all, probs_all)) if both else None
        ),
        "samples": len(labels_all),
        "confusion_matrix": confusion_matrix(
            labels_all, predictions, labels=[0, 1]
        ).tolist(),
    }


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    # Explicitly keep the frozen visual backbone in eval mode.
    model.visual.eval()

    total_loss = 0.0
    labels_all, probs_all = [], []

    for videos, labels in loader:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(videos)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        labels_all.extend(labels.int().detach().cpu().tolist())
        probs_all.extend(
            torch.sigmoid(logits).detach().cpu().tolist()
        )

    mean_loss = total_loss / max(1, len(labels_all))
    return calculate_metrics(labels_all, probs_all, mean_loss)


def main():
    parser = argparse.ArgumentParser(
        description="Train/evaluate the video-only LAV-DF baseline."
    )
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--val-samples", type=int, default=128)
    parser.add_argument("--test-samples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    if min(args.train_samples, args.val_samples, args.test_samples) < 2:
        parser.error("All sample limits must be at least 2.")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be at least 1.")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_data = VideoOnlyDataset("train")
    val_data = VideoOnlyDataset("dev")
    test_data = VideoOnlyDataset("test")

    train_indices = select_indices(train_data, args.train_samples, 42)
    val_indices = select_indices(val_data, args.val_samples, 43)
    test_indices = select_indices(test_data, args.test_samples, 44)

    print("Selected class counts:")
    print("Train:", class_counts(train_data, train_indices))
    print("Dev:", class_counts(val_data, val_indices))
    print("Test:", class_counts(test_data, test_indices))

    train_loader = DataLoader(
        Subset(train_data, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        Subset(val_data, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        Subset(test_data, test_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = VideoOnlyModel(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        list(model.temporal.parameters()) +
        list(model.classifier.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    checkpoint = ROOT / "checkpoints" / "video_only_v2_best.pt"
    output = ROOT / "results" / "video_only_v2_metrics.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint.exists() or output.exists():
        raise FileExistsError(
            "Version-2 output already exists. Rename these files first:\n"
            f"{checkpoint}\n{output}\n"
            "Existing files are never overwritten."
        )

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, device
        )

        print(f"\nEpoch {epoch}/{args.epochs}")
        print("Train:", json.dumps(train_metrics))
        print("Validation:", json.dumps(val_metrics))
        print("Seconds:", round(time.time() - started, 1))

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
        })

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), checkpoint)

    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    test_metrics = run_epoch(
        model, test_loader, criterion, device
    )

    report = {
        "model": "Video-only ResNet50 + BiLSTM",
        "label_definition": (
            "video_label = overall window fake-period overlap AND "
            "source metadata modify_video=True"
        ),
        "warning": (
            "Subset experiment; not full test-set performance."
            if args.test_samples < len(test_data)
            else "Full test split evaluated."
        ),
        "device": str(device),
        "train_samples": len(train_indices),
        "validation_samples": len(val_indices),
        "test_samples": len(test_indices),
        "train_class_counts": class_counts(train_data, train_indices),
        "validation_class_counts": class_counts(val_data, val_indices),
        "test_class_counts": class_counts(test_data, test_indices),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "history": history,
        "test": test_metrics,
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nVIDEO-ONLY TEST:")
    print(json.dumps(test_metrics, indent=2))
    print("Checkpoint:", checkpoint)
    print("Report:", output)


if __name__ == "__main__":
    main()
