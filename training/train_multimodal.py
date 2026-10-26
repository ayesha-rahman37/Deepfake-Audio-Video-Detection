"""Train the integrated LAV-DF detector with reproducible validation.

Smoke test:
  python training/train_multimodal.py --max-train-samples 8 --max-val-samples 4 --epochs 1 --batch-size 1 --no-pretrained-visual

Small pilot (not final research results):
  python training/train_multimodal.py --max-train-samples 512 --max-val-samples 128 --epochs 3 --batch-size 1

For an actual experiment, increase the subset only after profiling runtime and
GPU memory. Use the same settings/splits when comparing --fusion-mode concat
and --fusion-mode reliability.
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.lavdf_dataset import LAVDFDataset  # noqa: E402
from models.multimodal_model import MultimodalDeepfakeDetector  # noqa: E402


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_indices(dataset, max_samples: int, seed: int) -> list[int]:
    """Select a repeatable approximately balanced window subset.

    max_samples <= 0 means use the full split with its natural label balance.
    Subset selection is window-level, not video-level; full experiments should
    document video-level dependence when interpreting window metrics.
    """
    if max_samples <= 0 or max_samples >= len(dataset):
        return list(range(len(dataset)))

    by_label = {0: [], 1: []}
    for index, sample in enumerate(dataset.samples):
        by_label[int(sample["label"])].append(index)
    if not by_label[0] or not by_label[1]:
        raise RuntimeError("Both label classes are required for stratified selection")

    rng = random.Random(seed)
    for indices in by_label.values():
        rng.shuffle(indices)

    target_per_class = max_samples // 2
    selected = by_label[0][:target_per_class] + by_label[1][:target_per_class]
    selected_set = set(selected)

    # When one class is smaller than the requested balanced quota, fill the
    # remaining slots from unused windows instead of silently returning fewer.
    if max_samples % 2 or len(selected) < max_samples:
        remaining_candidates = [
            idx for idx, sample in enumerate(dataset.samples)
            if idx not in selected_set
        ]
        rng.shuffle(remaining_candidates)
        needed = max_samples - len(selected)
        selected.extend(remaining_candidates[:needed])

    rng.shuffle(selected)
    if len(selected) != max_samples:
        raise RuntimeError(
            f"Could not select {max_samples} samples; got {len(selected)}"
        )
    return selected


def make_loader(dataset, max_samples, batch_size, seed, shuffle):
    indices = stratified_indices(dataset, max_samples, seed)
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Stable default for Windows.
        pin_memory=torch.cuda.is_available(),
    )
    labels = [int(dataset.samples[i]["label"]) for i in indices]
    counts = Counter(labels)
    return loader, {"real": counts[0], "fake": counts[1]}


def calculate_metrics(labels: list[int], probabilities: list[float], loss: float) -> dict:
    predictions = [int(prob >= 0.5) for prob in probabilities]
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    both_classes = len(set(labels)) == 2
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)) if both_classes else None,
        "precision_fake": float(precision_score(labels, predictions, zero_division=0, pos_label=1)),
        "recall_fake": float(recall_score(labels, predictions, zero_division=0, pos_label=1)),
        "f1_fake": float(f1_score(labels, predictions, zero_division=0, pos_label=1)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if both_classes else None,
        "samples": len(labels),
        "confusion_matrix_labels_0_1": matrix.tolist(),
    }


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total = 0
    labels_all: list[int] = []
    probabilities_all: list[float] = []

    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        audio = batch["audio"].to(device, non_blocking=True)
        labels = batch["label"].float().to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(video, audio)
            logits = outputs["logits"]
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                trainable = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()

        count = int(labels.size(0))
        total_loss += float(loss.detach().item()) * count
        total += count
        labels_all.extend(labels.detach().cpu().int().tolist())
        probabilities_all.extend(torch.sigmoid(logits.detach()).cpu().tolist())

    if total == 0:
        raise RuntimeError("DataLoader returned no batches")
    return calculate_metrics(labels_all, probabilities_all, total_loss / total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-train-samples", type=int, default=64,
                        help="Maximum train windows; 0 means full train split")
    parser.add_argument("--max-val-samples", type=int, default=16,
                        help="Maximum validation windows; 0 means full dev split")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3,
                        help="Early-stopping patience on validation loss")
    parser.add_argument("--fusion-mode", choices=("concat", "reliability"), default="reliability")
    parser.add_argument("--no-pretrained-visual", action="store_true",
                        help="Skip pretrained visual weights; primarily for offline smoke tests")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch-size and patience must be >= 1")
    if args.max_train_samples == 1 or args.max_val_samples == 1:
        parser.error("Use at least 2 samples for a subset, or 0 for a full split")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_dataset = LAVDFDataset(split="train")
    val_dataset = LAVDFDataset(split="dev")
    train_loader, train_counts = make_loader(
        train_dataset, args.max_train_samples, args.batch_size, args.seed, True
    )
    val_loader, val_counts = make_loader(
        val_dataset, args.max_val_samples, args.batch_size, args.seed + 1, False
    )
    print(f"Train windows used: {len(train_loader.dataset)}; label counts: {train_counts}")
    print(f"Validation windows used: {len(val_loader.dataset)}; label counts: {val_counts}")

    model = MultimodalDeepfakeDetector(
        pretrained_visual=not args.no_pretrained_visual,
        freeze_visual=True,
        frame_batch_size=4,
        fusion_mode=args.fusion_mode,
    ).to(device)

    # Counter class imbalance for full/naturally distributed subsets.
    positive_count = max(1, train_counts["fake"])
    negative_count = max(1, train_counts["real"])
    pos_weight = torch.tensor([negative_count / positive_count], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-6
    )

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    results_dir = PROJECT_ROOT / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"multimodal_{args.fusion_mode}_best.pt"
    result_path = results_dir / f"multimodal_{args.fusion_mode}_training_metrics.json"
    history = []
    best_val_loss = float("inf")
    best_epoch = None
    stale_epochs = 0
    start_all = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, device, criterion, optimizer)
        val_metrics = run_epoch(model, val_loader, device, criterion)
        scheduler.step(val_metrics["loss"])
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "validation": val_metrics,
            "epoch_seconds": round(time.perf_counter() - epoch_start, 2),
        }
        history.append(row)
        print(f"\nEpoch {epoch}/{args.epochs} ({row['epoch_seconds']} s)")
        print("Learning rate:", row["learning_rate"])
        print("Train:", json.dumps(train_metrics, indent=2))
        print("Validation:", json.dumps(val_metrics, indent=2))

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "validation_metrics": val_metrics,
                "fusion_mode": args.fusion_mode,
                "args": vars(args),
                "class_counts": {"train": train_counts, "validation": val_counts},
                "pos_weight": float(pos_weight.item()),
            }, checkpoint_path)
            print(f"Saved best checkpoint: {checkpoint_path}")
        else:
            stale_epochs += 1

        with result_path.open("w", encoding="utf-8") as file:
            json.dump({
                "warning": "Subset runs are pilot results, not final test performance.",
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "fusion_mode": args.fusion_mode,
                "args": vars(args),
                "class_counts": {"train": train_counts, "validation": val_counts},
                "best_epoch_so_far": best_epoch,
                "history": history,
            }, file, indent=2)

        if stale_epochs >= args.patience:
            print(f"Early stopping: validation loss did not improve for {args.patience} epochs.")
            break

    print(f"Best epoch: {best_epoch}; best validation loss: {best_val_loss:.6f}")
    print(f"Total run time: {time.perf_counter() - start_all:.1f} seconds")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved history: {result_path}")
    print("Training complete. Do not report these metrics as final until held-out test evaluation.")


if __name__ == "__main__":
    main()
