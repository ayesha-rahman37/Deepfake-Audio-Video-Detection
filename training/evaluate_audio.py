import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.audio_encoder import AudioEncoder
from training.train_audio import (
    DEFAULT_CHECKPOINT, build_real_dataset_class, normalize,
    collate_audio_batch, DummyAudioDataset,
)
from utils.device import device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if args.max_samples < 0 or args.batch_size < 1:
        parser.error("--max_samples must be >=0 and --batch_size >=1")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    dataset = DummyAudioDataset(32) if args.dummy else build_real_dataset_class()(args.split)
    indices = list(range(len(dataset)))
    if args.max_samples > 0 and len(indices) > args.max_samples:
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(dataset), generator=generator)[:args.max_samples].tolist()

    loader = DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_audio_batch,
    )
    model = AudioEncoder().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    probabilities, labels = [], []
    skipped_batches = 0
    with torch.inference_mode():
        for batch in loader:
            if batch is None:
                skipped_batches += 1
                continue
            audio, label = batch
            _, logits = model(normalize(audio).to(device))
            probabilities.extend(torch.sigmoid(logits).reshape(-1).cpu().tolist())
            labels.extend(label.int().reshape(-1).cpu().tolist())

    if not labels:
        raise RuntimeError("No valid audio samples available for evaluation.")

    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    pred = (p >= 0.5).astype(np.int64)
    both = len(np.unique(y)) == 2
    report = {
        "split": "dummy" if args.dummy else args.split,
        "checkpoint": str(checkpoint.resolve()),
        "samples_requested": len(indices),
        "samples_evaluated": len(labels),
        "skipped_batches": skipped_batches,
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if both else None,
        "precision_fake": float(precision_score(y, pred, zero_division=0)),
        "recall_fake": float(recall_score(y, pred, zero_division=0)),
        "f1_fake": float(f1_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "confusion_matrix_labels_0_1": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "note": "Audio-only dataset sampling protocol may differ from multimodal comparison; compare only when sample IDs/windows are verified identical.",
    }
    print(json.dumps(report, indent=2))

    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = PROJECT_ROOT / output
    else:
        split_name = "dummy" if args.dummy else args.split
        size_name = "full" if args.max_samples == 0 else str(len(indices))
        output = PROJECT_ROOT / "results" / f"audio_{split_name}_{size_name}_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing report: {output}. "
            "Use --output with a new filename."
        )
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved: {output}")


if __name__ == "__main__":
    main()
