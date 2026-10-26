"""Evaluate a trained multimodal checkpoint on the held-out LAV-DF test split.

Pilot only:
  python evaluation/evaluate_multimodal.py --max-test-samples 8 --no-pretrained-visual

Final evaluation after selecting the model on validation:
  python evaluation/evaluate_multimodal.py --max-test-samples 0
"""

import argparse
import json
import sys
from pathlib import Path

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.lavdf_dataset import LAVDFDataset  # noqa: E402
from models.multimodal_model import MultimodalDeepfakeDetector  # noqa: E402
from training.train_multimodal import make_loader, seed_everything  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Checkpoint path; default follows --fusion-mode")
    parser.add_argument("--fusion-mode", choices=("concat", "reliability"), default="reliability")
    parser.add_argument("--max-test-samples", type=int, default=0,
                        help="Maximum stratified test windows; 0 means the entire test split")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_test_samples == 1:
        parser.error("batch-size must be >=1; test samples must be >=2 or 0 for full test split")

    seed_everything(args.seed)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (
        PROJECT_ROOT / "checkpoints" / f"multimodal_{args.fusion_mode}_best.pt"
    )
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_mode = checkpoint.get("fusion_mode", args.fusion_mode)
    if saved_mode != args.fusion_mode:
        raise ValueError(
            f"Checkpoint fusion_mode={saved_mode!r}, but --fusion-mode={args.fusion_mode!r}"
        )

    model = MultimodalDeepfakeDetector(
        # The checkpoint contains the trained visual encoder weights; avoid
        # downloading ImageNet weights before loading the complete state dict.
        pretrained_visual=False,
        freeze_visual=True,
        frame_batch_size=4,
        fusion_mode=saved_mode,
    ).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture does not match the current model code. "
            "Train a fresh checkpoint with the current fusion implementation."
        ) from exc
    model.eval()

    test_dataset = LAVDFDataset(split="test")
    loader, class_counts = make_loader(
        test_dataset, args.max_test_samples, args.batch_size, args.seed, False
    )
    y_true, y_prob = [], []
    total_loss, total = 0.0, 0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(device, non_blocking=True)
            audio = batch["audio"].to(device, non_blocking=True)
            labels = batch["label"].float().to(device, non_blocking=True)
            output = model(video, audio)
            logits = output["logits"]
            loss = criterion(logits, labels)
            n = int(labels.size(0))
            total_loss += float(loss.item()) * n
            total += n
            y_true.extend(labels.cpu().int().tolist())
            y_prob.extend(torch.sigmoid(logits).cpu().tolist())

    y_pred = [int(p >= 0.5) for p in y_prob]
    both = len(set(y_true)) == 2
    metrics = {
        "split": "test",
        "fusion_mode": saved_mode,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "samples": total,
        "class_counts": class_counts,
        "loss_unweighted": total_loss / max(total, 1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if both else None,
        "precision_fake": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both else None,
        "confusion_matrix_labels_0_1": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "note": "Interpret subset evaluation as pilot only; final metrics require the full fixed test split.",
    }
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"multimodal_{saved_mode}_test_metrics.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved test metrics: {output_path}")


if __name__ == "__main__":
    main()
