"""Temporal-order ablation for the LAV-DF multimodal detector.

Compares normal frame order with reversed frame order on the exact same
window-level samples. Audio is held fixed. This is a diagnostic of sensitivity
to frame ordering, not a retrained model or temporal-localization experiment.

Example:
    python evaluation/temporal_order_ablation.py --split test --samples 256
    python evaluation/temporal_order_ablation.py --split dev --samples 256
    Use --samples 0 for the complete split (computationally expensive).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.lavdf_dataset import LAVDFDataset
from models.multimodal_model import MultimodalDeepfakeDetector


def choose_indices(dataset, limit, seed):
    if limit == 0 or limit >= len(dataset):
        return list(range(len(dataset)))
    groups = {0: [], 1: []}
    for index, row in enumerate(dataset.samples):
        groups[int(row["label"])].append(index)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    n0 = min(len(groups[0]), limit // 2)
    n1 = min(len(groups[1]), limit - n0)
    selected = groups[0][:n0] + groups[1][:n1]
    if len(selected) < limit:
        used = set(selected)
        remaining = [i for i in range(len(dataset)) if i not in used]
        rng.shuffle(remaining)
        selected.extend(remaining[:limit - len(selected)])
    rng.shuffle(selected)
    return selected


def calculate_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    both_classes = len(np.unique(labels)) == 2
    return {
        "samples": int(len(labels)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(labels, predictions))
            if both_classes else None
        ),
        "precision_fake": float(precision_score(labels, predictions, zero_division=0)),
        "recall_fake": float(recall_score(labels, predictions, zero_division=0)),
        "f1_fake": float(f1_score(labels, predictions, zero_division=0)),
        "macro_f1": float(f1_score(
            labels, predictions, average="macro", zero_division=0
        )),
        "roc_auc": (
            float(roc_auc_score(labels, probabilities)) if both_classes else None
        ),
        "confusion_matrix_labels_0_1": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--samples", type=int, default=256,
                        help="Balanced subset size; 0 evaluates the full split.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--checkpoint", default="checkpoints/multimodal_reliability_best.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if args.samples < 0 or args.batch_size < 1:
        parser.error("--samples must be >= 0 and --batch-size must be >= 1")
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    fusion_mode = checkpoint.get("fusion_mode", "reliability")
    if fusion_mode != "reliability":
        raise ValueError(
            f"Expected reliability checkpoint, found fusion_mode={fusion_mode!r}"
        )

    dataset = LAVDFDataset(split=args.split)
    selected = choose_indices(dataset, args.samples, args.seed)
    labels_selected = [int(dataset.samples[i]["label"]) for i in selected]
    print("Split:", args.split)
    print("Selected windows:", len(selected))
    print("Class counts:", {
        "real": labels_selected.count(0), "fake": labels_selected.count(1)
    })

    loader = DataLoader(
        Subset(dataset, selected),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    model = MultimodalDeepfakeDetector(
        pretrained_visual=False,
        freeze_visual=True,
        frame_batch_size=4,
        fusion_mode="reliability",
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    y_all, normal_probs, reversed_probs = [], [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            video = batch["video"].to(device, non_blocking=True)
            audio = batch["audio"].to(device, non_blocking=True)
            labels = batch["label"].long().reshape(-1)

            normal_logits = model(video, audio)["logits"].reshape(-1)
            reversed_logits = model(torch.flip(video, dims=[1]), audio)["logits"].reshape(-1)

            y_all.extend(labels.tolist())
            normal_probs.extend(torch.sigmoid(normal_logits).cpu().tolist())
            reversed_probs.extend(torch.sigmoid(reversed_logits).cpu().tolist())

            if batch_index % 25 == 0 or batch_index == len(loader):
                print(f"Processed {len(y_all)}/{len(selected)} windows", flush=True)

    y = np.asarray(y_all, dtype=np.int64)
    p_normal = np.asarray(normal_probs, dtype=np.float64)
    p_reversed = np.asarray(reversed_probs, dtype=np.float64)
    normal_pred = p_normal >= args.threshold
    reversed_pred = p_reversed >= args.threshold

    result = {
        "experiment": "temporal_order_ablation",
        "split": args.split,
        "fusion_mode": fusion_mode,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "seed": args.seed,
        "sample_count": int(len(y)),
        "class_counts": {"real": int((y == 0).sum()), "fake": int((y == 1).sum())},
        "threshold": args.threshold,
        "protocol": (
            "Same windows and labels; compare original 16-frame sequence with "
            "the exact reversed sequence; audio is unchanged. No retraining."
        ),
        "normal_order": calculate_metrics(y, p_normal, args.threshold),
        "reversed_order": calculate_metrics(y, p_reversed, args.threshold),
        "sensitivity": {
            "mean_absolute_fake_score_change": float(np.mean(np.abs(p_normal - p_reversed))),
            "median_absolute_fake_score_change": float(np.median(np.abs(p_normal - p_reversed))),
            "prediction_flip_count": int(np.sum(normal_pred != reversed_pred)),
            "prediction_flip_rate": float(np.mean(normal_pred != reversed_pred)),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "interpretation_note": (
            "A difference indicates sensitivity to temporal order under reversal; "
            "no difference does not prove that the temporal branch is unused. "
            "This is a subset diagnostic unless sample_count equals the full split."
        ),
    }

    output = Path(args.output) if args.output else (
        ROOT / "results" / f"temporal_order_ablation_{args.split}_{len(y)}.json"
    )
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite {output}; provide a new --output filename."
        )
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\nTEMPORAL ORDER ABLATION")
    print(json.dumps(result, indent=2))
    print("\nSaved:", output)


if __name__ == "__main__":
    main()
