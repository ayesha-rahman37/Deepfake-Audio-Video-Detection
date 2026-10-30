
"""Resumable full-test evaluation for the multimodal LAV-DF model."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.lavdf_dataset import LAVDFDataset
from models.multimodal_model import MultimodalDeepfakeDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--fusion-mode", choices=("concat", "reliability"),
                        default="reliability")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--parts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--chunks-per-run", type=int, default=1)
    args = parser.parse_args()

    if args.batch_size < 1 or args.parts < 1 or args.chunks_per_run < 1:
        parser.error("batch-size, parts and chunks-per-run must be >= 1")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (
        ROOT / "checkpoints" / f"multimodal_{args.fusion_mode}_best.pt"
    )
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mode = checkpoint.get("fusion_mode", args.fusion_mode)
    if mode != args.fusion_mode:
        raise ValueError(f"Checkpoint mode is {mode}, requested {args.fusion_mode}")

    model = MultimodalDeepfakeDetector(
        pretrained_visual=False, freeze_visual=True,
        frame_batch_size=4, fusion_mode=mode,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    dataset = LAVDFDataset(split="test")
    total_samples = len(dataset)
    output_dir = ROOT / "results" / f"reliability_chunks_{mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prevent accidentally reusing chunks from a different checkpoint or setup.
    checkpoint_stat = checkpoint_path.stat()
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime": checkpoint_stat.st_mtime,
        "fusion_mode": mode,
        "dataset_samples": total_samples,
        "parts": args.parts,
        "seed": args.seed,
    }

    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old != manifest:
            raise RuntimeError(
                "Saved chunks do not match this checkpoint/settings. "
                f"Back up or remove this folder before starting fresh: {output_dir}"
            )
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Deterministic, non-overlapping chunks cover every test window exactly once.
    boundaries = np.linspace(0, total_samples, args.parts + 1, dtype=int)
    completed_this_run = 0

    for part in range(args.parts):
        chunk_path = output_dir / f"part_{part + 1:02d}.npz"
        if chunk_path.exists():
            print(f"Part {part + 1}/{args.parts}: already saved; skipping.")
            continue

        if completed_this_run >= args.chunks_per_run:
            break

        start, end = int(boundaries[part]), int(boundaries[part + 1])
        loader = DataLoader(
            Subset(dataset, list(range(start, end))),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        labels_all, probs_all = [], []
        loss_sum = 0.0
        count = 0

        print(f"\nEvaluating part {part + 1}/{args.parts}: "
              f"samples {start + 1}-{end} ({end - start} samples)")
        with torch.inference_mode():
            for batch in loader:
                video = batch["video"].to(device, non_blocking=True)
                audio = batch["audio"].to(device, non_blocking=True)
                labels = batch["label"].float().to(device, non_blocking=True)

                output = model(video, audio)
                logits = output["logits"].reshape(-1)
                loss_sum += float(
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, labels.reshape(-1), reduction="sum"
                    ).item()
                )
                probs = torch.sigmoid(logits)
                labels_all.extend(labels.cpu().int().reshape(-1).tolist())
                probs_all.extend(probs.cpu().float().reshape(-1).tolist())
                count += int(labels.numel())

        # Write to a temporary file first, so interrupted writes are not mistaken
        # for completed parts.
        temp_path = chunk_path.with_suffix(".tmp.npz")
        np.savez(
            temp_path,
            labels=np.asarray(labels_all, dtype=np.int8),
            probabilities=np.asarray(probs_all, dtype=np.float32),
            loss_sum=np.asarray(loss_sum, dtype=np.float64),
            count=np.asarray(count, dtype=np.int64),
            start=np.asarray(start, dtype=np.int64),
            end=np.asarray(end, dtype=np.int64),
        )
        temp_path.replace(chunk_path)
        completed_this_run += 1
        print(f"Saved part {part + 1}/{args.parts} ({count} samples).")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    chunk_files = [output_dir / f"part_{p + 1:02d}.npz"
                   for p in range(args.parts)]
    if not all(path.exists() for path in chunk_files):
        remaining = sum(not path.exists() for path in chunk_files)
        print(f"\nPaused safely. {remaining} part(s) remain.")
        print("Let the laptop cool if needed, then run the same command again.")
        return

    # Finalize from saved predictions only; the model is not run again here.
    labels_parts, probs_parts = [], []
    total_loss_sum = 0.0
    total_count = 0

    for part, path in enumerate(chunk_files):
        with np.load(path) as data:
            expected_start, expected_end = (
                int(boundaries[part]), int(boundaries[part + 1])
            )
            if int(data["start"]) != expected_start or int(data["end"]) != expected_end:
                raise RuntimeError(f"Incorrect sample range in {path}")
            labels_parts.append(data["labels"].copy())
            probs_parts.append(data["probabilities"].copy())
            total_loss_sum += float(data["loss_sum"])
            total_count += int(data["count"])

    y_true = np.concatenate(labels_parts)
    y_prob = np.concatenate(probs_parts)
    if total_count != total_samples or len(y_true) != total_samples:
        raise RuntimeError(
            f"Expected {total_samples} samples but found {total_count}; "
            "refusing to report final metrics."
        )

    y_pred = (y_prob >= 0.5).astype(np.int8)
    both = len(np.unique(y_true)) == 2
    metrics = {
        "split": "test",
        "fusion_mode": mode,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "samples": total_count,
        "parts": args.parts,
        "class_counts": {
            "real": int((y_true == 0).sum()),
            "fake": int((y_true == 1).sum()),
        },
        "loss_unweighted": total_loss_sum / total_count,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if both else None,
        "precision_fake": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both else None,
        "confusion_matrix_labels_0_1": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
        "note": "Full fixed test split; predictions combined from saved non-overlapping chunks.",
    }

    # Preserve the previous 1024-sample pilot JSON; write final results separately.
    final_path = ROOT / "results" / f"multimodal_{mode}_test_metrics_full.json"
    final_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nFULL TEST RESULT")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved final metrics: {final_path}")
    print(f"Saved chunk predictions: {output_dir}")


if __name__ == "__main__":
    main()
