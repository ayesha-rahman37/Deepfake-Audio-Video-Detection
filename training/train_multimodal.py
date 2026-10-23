"""Train and evaluate the integrated LAV-DF audio-video detector.

Quick smoke run (real dataset windows, one epoch):
python training/train_multimodal.py --max-train-samples 8 --max-val-samples 4 --epochs 1 --batch-size 1

Use larger limits for actual experiments only after the smoke run passes.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.lavdf_dataset import LAVDFDataset  # noqa: E402
from models.multimodal_model import MultimodalDeepfakeDetector  # noqa: E402


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_indices(dataset, max_samples: int, seed: int):
    """Select a reproducible, approximately balanced subset of dataset windows."""
    if max_samples <= 0 or max_samples >= len(dataset):
        return list(range(len(dataset)))

    by_label = {0: [], 1: []}
    for index, item in enumerate(dataset.samples):
        by_label[int(item["label"])].append(index)

    rng = random.Random(seed)
    if not by_label[0] or not by_label[1]:
        raise RuntimeError("Dataset subset selection requires both labels 0 and 1")

    target_per_class = max_samples // 2
    selected = []
    for label in (0, 1):
        rng.shuffle(by_label[label])
        selected.extend(by_label[label][:target_per_class])

    if max_samples % 2:
        remaining_label = 0 if len(by_label[0]) > target_per_class else 1
        selected.append(by_label[remaining_label][target_per_class])

    rng.shuffle(selected)
    if len(selected) < 2:
        raise ValueError("Use --max-train-samples of at least 2")
    return selected


def make_loader(dataset, max_samples, batch_size, seed, shuffle):
    indices = stratified_indices(dataset, max_samples, seed)
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Reliable default for Windows/VS Code.
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total = 0
    labels_all, probs_all = [], []

    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        audio = batch["audio"].to(device, non_blocking=True)
        labels = batch["label"].float().to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            output = model(video, audio)
            logits = output["logits"]
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()

        batch_count = labels.size(0)
        total_loss += float(loss.detach().item()) * batch_count
        total += batch_count
        labels_all.extend(labels.detach().cpu().int().tolist())
        probs_all.extend(torch.sigmoid(logits.detach()).cpu().tolist())

    if total == 0:
        raise RuntimeError("DataLoader returned no batches")

    predictions = [int(prob >= 0.5) for prob in probs_all]
    metrics = {
        "loss": total_loss / total,
        "accuracy": accuracy_score(labels_all, predictions),
        "precision": precision_score(labels_all, predictions, zero_division=0),
        "recall": recall_score(labels_all, predictions, zero_division=0),
        "f1": f1_score(labels_all, predictions, zero_division=0),
        "samples": total,
    }
    metrics["roc_auc"] = (
        roc_auc_score(labels_all, probs_all) if len(set(labels_all)) == 2 else None
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-samples", type=int, default=64,
                        help="Maximum stratified train windows; 0 means the full split")
    parser.add_argument("--max-val-samples", type=int, default=16,
                        help="Maximum stratified dev windows; 0 means the full split")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained-visual", action="store_true",
                        help="Do not load ImageNet weights (useful for offline smoke tests)")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be at least 1")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_dataset = LAVDFDataset(split="train")
    val_dataset = LAVDFDataset(split="dev")
    train_loader = make_loader(
        train_dataset, args.max_train_samples, args.batch_size, args.seed, True
    )
    val_loader = make_loader(
        val_dataset, args.max_val_samples, args.batch_size, args.seed + 1, False
    )
    print(f"Train windows used: {len(train_loader.dataset)}")
    print(f"Validation windows used: {len(val_loader.dataset)}")

    model = MultimodalDeepfakeDetector(
        pretrained_visual=not args.no_pretrained_visual,
        freeze_visual=True,
        frame_batch_size=4,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    criterion = nn.BCEWithLogitsLoss()

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    results_dir = PROJECT_ROOT / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "multimodal_best.pt"
    history = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, criterion, optimizer)
        val_metrics = run_epoch(model, val_loader, device, criterion)
        row = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(row)
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("Train:", json.dumps(train_metrics, indent=2))
        print("Validation:", json.dumps(val_metrics, indent=2))

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "validation_metrics": val_metrics,
                "args": vars(args),
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    result_path = results_dir / "multimodal_training_metrics.json"
    with result_path.open("w", encoding="utf-8") as file:
        json.dump({
            "warning": "Small smoke-test subsets are not publishable final evaluation results.",
            "device": str(device),
            "history": history,
        }, file, indent=2)
    print(f"Saved metrics: {result_path}")
    print("Multimodal training run completed.")


if __name__ == "__main__":
    main()
