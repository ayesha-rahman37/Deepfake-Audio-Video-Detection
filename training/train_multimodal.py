
"""Train the LAV-DF multimodal deepfake detector.

Pilot:
    python training/train_multimodal.py --max-train-samples 512 --max-val-samples 128 --epochs 3 --batch-size 1

Smoke test (random visual weights; not for research results):
    python training/train_multimodal.py --max-train-samples 8 --max-val-samples 4 --epochs 1 --batch-size 1 --no-pretrained-visual
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

from datasets.lavdf_dataset import LAVDFDataset
from models.multimodal_model import MultimodalDeepfakeDetector


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_indices(dataset, max_samples, seed):
    """Select a reproducible, approximately balanced subset."""

    if max_samples <= 0 or max_samples >= len(dataset):
        return list(range(len(dataset)))

    by_label = {0: [], 1: []}

    for index, sample in enumerate(dataset.samples):
        label = int(sample["label"])

        if label not in by_label:
            raise ValueError(f"Unexpected label: {label}")

        by_label[label].append(index)

    if not by_label[0] or not by_label[1]:
        raise RuntimeError("Both real and fake samples are required.")

    rng = random.Random(seed)

    for indices in by_label.values():
        rng.shuffle(indices)

    target_per_class = max_samples // 2

    selected = (
        by_label[0][:target_per_class]
        + by_label[1][:target_per_class]
    )

    selected_set = set(selected)

    if len(selected) < max_samples:
        remaining = [
            index
            for index in range(len(dataset))
            if index not in selected_set
        ]

        rng.shuffle(remaining)

        selected.extend(
            remaining[:max_samples - len(selected)]
        )

    rng.shuffle(selected)

    if len(selected) != max_samples:
        raise RuntimeError("Could not create the requested subset.")

    return selected


def make_loader(dataset, max_samples, batch_size, seed, shuffle):
    indices = stratified_indices(
        dataset,
        max_samples,
        seed,
    )

    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    counts = Counter(
        int(dataset.samples[index]["label"])
        for index in indices
    )

    return loader, {
        "real": counts[0],
        "fake": counts[1],
    }


def calculate_metrics(labels, probabilities, loss):
    predictions = [
        int(probability >= 0.5)
        for probability in probabilities
    ]

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    both_classes = len(set(labels)) == 2

    return {
        "loss": float(loss),
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "balanced_accuracy": (
            float(balanced_accuracy_score(labels, predictions))
            if both_classes else None
        ),
        "precision_fake": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
                pos_label=1,
            )
        ),
        "recall_fake": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
                pos_label=1,
            )
        ),
        "f1_fake": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
                pos_label=1,
            )
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "roc_auc": (
            float(roc_auc_score(labels, probabilities))
            if both_classes else None
        ),
        "samples": len(labels),
        "confusion_matrix_labels_0_1": matrix.tolist(),
    }


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None

    model.train(training)

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_probabilities = []

    for batch in loader:
        video = batch["video"].to(
            device,
            non_blocking=True,
        )

        audio = batch["audio"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].float().to(
            device,
            non_blocking=True,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(video, audio)

            logits = outputs["logits"]

            loss = criterion(logits, labels)

            if training:
                loss.backward()

                trainable_parameters = [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                    and parameter.grad is not None
                ]

                if trainable_parameters:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        max_norm=1.0,
                    )

                optimizer.step()

        batch_size = int(labels.size(0))

        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size

        all_labels.extend(
            labels.detach().cpu().int().tolist()
        )

        all_probabilities.extend(
            torch.sigmoid(logits.detach()).cpu().tolist()
        )

    if total_samples == 0:
        raise RuntimeError("DataLoader returned no samples.")

    return calculate_metrics(
        all_labels,
        all_probabilities,
        total_loss / total_samples,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=512,
        help="Maximum training windows; 0 means full training split.",
    )

    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=128,
        help="Maximum validation windows; 0 means full validation split.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--fusion-mode",
        choices=("concat", "reliability"),
        default="reliability",
    )

    parser.add_argument(
        "--no-pretrained-visual",
        action="store_true",
        help="Disable pretrained visual weights. Use only for smoke tests.",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("epochs must be >= 1")

    if args.batch_size < 1:
        parser.error("batch-size must be >= 1")

    if args.patience < 1:
        parser.error("patience must be >= 1")

    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")

    if args.max_train_samples == 1 or args.max_val_samples == 1:
        parser.error("Subset size must be 0 or at least 2.")

    if args.no_pretrained_visual:
        print(
            "\nWARNING: Random visual initialization enabled. "
            "Do not use this run as final research training.\n"
        )

    seed_everything(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("MULTIMODAL DEEPFAKE DETECTOR TRAINING")
    print("=" * 65)

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Fusion mode:", args.fusion_mode)
    print("Pretrained visual weights:", not args.no_pretrained_visual)

    train_dataset = LAVDFDataset(split="train")
    val_dataset = LAVDFDataset(split="dev")

    print("\nAvailable dataset windows:")
    print("Training:", len(train_dataset))
    print("Validation:", len(val_dataset))

    train_loader, train_counts = make_loader(
        train_dataset,
        args.max_train_samples,
        args.batch_size,
        args.seed,
        True,
    )

    val_loader, val_counts = make_loader(
        val_dataset,
        args.max_val_samples,
        args.batch_size,
        args.seed + 1,
        False,
    )

    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        raise RuntimeError("Training or validation dataset is empty.")

    print("\nSelected samples:")
    print("Training:", len(train_loader.dataset), train_counts)
    print("Validation:", len(val_loader.dataset), val_counts)

    model = MultimodalDeepfakeDetector(
        pretrained_visual=not args.no_pretrained_visual,
        freeze_visual=True,
        frame_batch_size=4,
        fusion_mode=args.fusion_mode,
    ).to(device)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("\nModel parameters:")
    print(f"Total: {total_parameters:,}")
    print(f"Trainable: {trainable_parameters:,}")
    print(f"Frozen: {total_parameters - trainable_parameters:,}")

    if trainable_parameters == 0:
        raise RuntimeError("The model has no trainable parameters.")

    positive_count = max(1, train_counts["fake"])
    negative_count = max(1, train_counts["real"])

    pos_weight = torch.tensor(
        [negative_count / positive_count],
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
    )

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    results_dir = PROJECT_ROOT / "results"

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Candidate names protect the previous best checkpoint.
    checkpoint_path = (
        checkpoint_dir
        / f"multimodal_{args.fusion_mode}_candidate.pt"
    )

    result_path = (
        results_dir
        / f"multimodal_{args.fusion_mode}_candidate_metrics.json"
    )

    if checkpoint_path.exists():
        raise FileExistsError(
            f"Candidate checkpoint already exists: {checkpoint_path}\n"
            "Rename or back up that candidate before starting another run."
        )

    history = []

    best_val_loss = float("inf")
    best_epoch = None
    stale_epochs = 0

    start_all = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()

        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            criterion,
        )

        scheduler.step(val_metrics["loss"])

        row = {
            "epoch": epoch,
            "learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
            "train": train_metrics,
            "validation": val_metrics,
            "epoch_seconds": round(
                time.perf_counter() - epoch_start,
                2,
            ),
        }

        history.append(row)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print("Time:", row["epoch_seconds"], "seconds")
        print("Learning rate:", row["learning_rate"])

        print(
            "Train:",
            json.dumps(train_metrics, indent=2),
        )

        print(
            "Validation:",
            json.dumps(val_metrics, indent=2),
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            stale_epochs = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": val_metrics,
                    "fusion_mode": args.fusion_mode,
                    "args": vars(args),
                    "class_counts": {
                        "train": train_counts,
                        "validation": val_counts,
                    },
                    "pos_weight": float(pos_weight.item()),
                    "total_parameters": total_parameters,
                    "trainable_parameters": trainable_parameters,
                    "pretrained_visual": not args.no_pretrained_visual,
                },
                checkpoint_path,
            )

            print("Saved candidate checkpoint:", checkpoint_path)

        else:
            stale_epochs += 1

        with result_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "warning": (
                        "Pilot results are not final held-out test performance."
                    ),
                    "device": str(device),
                    "gpu_name": (
                        torch.cuda.get_device_name(0)
                        if device.type == "cuda"
                        else None
                    ),
                    "fusion_mode": args.fusion_mode,
                    "args": vars(args),
                    "class_counts": {
                        "train": train_counts,
                        "validation": val_counts,
                    },
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_val_loss,
                    "history": history,
                },
                file,
                indent=2,
            )

        if stale_epochs >= args.patience:
            print(
                "\nEarly stopping: validation loss did not improve."
            )
            break

    print("\n" + "=" * 65)
    print("TRAINING FINISHED")
    print("=" * 65)

    print("Best epoch:", best_epoch)
    print("Best validation loss:", best_val_loss)
    print(
        "Total runtime:",
        round(time.perf_counter() - start_all, 2),
        "seconds",
    )

    print("Candidate checkpoint:", checkpoint_path)
    print("Training history:", result_path)

    print(
        "\nIMPORTANT: This is a candidate checkpoint. "
        "Do not replace the existing production checkpoint "
        "until evaluation is complete."
    )


if __name__ == "__main__":
    main()
