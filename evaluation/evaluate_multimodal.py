"""Resumable multimodal LAV-DF evaluation on dev/test splits."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from datasets.lavdf_dataset import LAVDFDataset
from models.multimodal_model import MultimodalDeepfakeDetector


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(np.int8)
    both = len(np.unique(y)) == 2
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if both else None,
        "precision_fake": float(precision_score(y, pred, zero_division=0)),
        "recall_fake": float(recall_score(y, pred, zero_division=0)),
        "f1_fake": float(f1_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "confusion_matrix_labels_0_1": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--fusion-mode", choices=("concat", "reliability"), default="reliability")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--parts", type=int, default=56)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--chunks-per-run", type=int, default=56)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    if min(args.batch_size, args.parts, args.chunks_per_run) < 1:
        parser.error("batch-size, parts and chunks-per-run must be >= 1")
    if args.threshold is not None and not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint) if args.checkpoint else ROOT / "checkpoints" / f"multimodal_{args.fusion_mode}_best.pt"
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    mode = checkpoint.get("fusion_mode", args.fusion_mode)
    if mode != args.fusion_mode:
        raise ValueError(f"Checkpoint mode is {mode}, requested {args.fusion_mode}")

    model = MultimodalDeepfakeDetector(
        pretrained_visual=False, freeze_visual=True,
        frame_batch_size=4, fusion_mode=mode
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    dataset = LAVDFDataset(split=args.split)
    total = len(dataset)
    # Preserve the original test chunk folder and all its existing files.
    out_dir = ROOT / "results" / (
        f"reliability_chunks_{mode}" if args.split == "test"
        else f"reliability_chunks_dev_{mode}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    st = ckpt_path.stat()
    manifest = {
        "checkpoint": str(ckpt_path.resolve()),
        "checkpoint_size": st.st_size,
        "checkpoint_mtime": st.st_mtime,
        "fusion_mode": mode,
        "dataset_samples": total,
        "parts": args.parts,
        "seed": args.seed,
    }
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old != manifest:
            raise RuntimeError(f"Chunk manifest mismatch; existing files preserved: {out_dir}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    bounds = np.linspace(0, total, args.parts + 1, dtype=int)
    done = 0
    for part in range(args.parts):
        chunk = out_dir / f"part_{part + 1:02d}.npz"
        if chunk.exists():
            print(f"Part {part + 1}/{args.parts}: already saved; skipping.")
            continue
        if done >= args.chunks_per_run:
            break
        start, end = int(bounds[part]), int(bounds[part + 1])
        loader = DataLoader(
            Subset(dataset, list(range(start, end))),
            batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=(device.type == "cuda")
        )
        ys, ps, loss_sum, count = [], [], 0.0, 0
        print(f"\nEvaluating {args.split} part {part + 1}/{args.parts}: {start + 1}-{end}")
        with torch.inference_mode():
            for bi, batch in enumerate(loader, start=1):
                video = batch["video"].to(device, non_blocking=True)
                audio = batch["audio"].to(device, non_blocking=True)
                labels = batch["label"].float().to(device, non_blocking=True)
                logits = model(video, audio)["logits"].reshape(-1)
                loss_sum += float(torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels.reshape(-1), reduction="sum").item())
                ys.extend(labels.cpu().int().reshape(-1).tolist())
                ps.extend(torch.sigmoid(logits).cpu().float().reshape(-1).tolist())
                count += int(labels.numel())
                if bi % 100 == 0:
                    print(f"  processed {count}/{end-start}", flush=True)
        tmp = chunk.with_suffix(".tmp.npz")
        np.savez(tmp, labels=np.asarray(ys, dtype=np.int8),
                 probabilities=np.asarray(ps, dtype=np.float32),
                 loss_sum=np.asarray(loss_sum, dtype=np.float64),
                 count=np.asarray(count, dtype=np.int64),
                 start=np.asarray(start, dtype=np.int64),
                 end=np.asarray(end, dtype=np.int64))
        tmp.replace(chunk)
        done += 1
        print(f"Saved part {part + 1}/{args.parts} ({count} samples).")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    chunk_paths = [out_dir / f"part_{i+1:02d}.npz" for i in range(args.parts)]
    if not all(p.exists() for p in chunk_paths):
        print(f"\nPaused safely. {sum(not p.exists() for p in chunk_paths)} part(s) remain. Rerun same command to resume.")
        return

    y_parts, p_parts, total_loss, count = [], [], 0.0, 0
    for i, path in enumerate(chunk_paths):
        with np.load(path) as d:
            if int(d["start"]) != int(bounds[i]) or int(d["end"]) != int(bounds[i+1]):
                raise RuntimeError(f"Incorrect sample range in {path}")
            y_parts.append(d["labels"].copy())
            p_parts.append(d["probabilities"].copy())
            total_loss += float(d["loss_sum"])
            count += int(d["count"])
    y, p = np.concatenate(y_parts), np.concatenate(p_parts)
    if count != total or len(y) != total:
        raise RuntimeError(f"Expected {total} samples, found {count}; refusing final metrics.")

    result = {
        "split": args.split, "fusion_mode": mode,
        "checkpoint": str(ckpt_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "samples": count, "parts": args.parts,
        "class_counts": {"real": int((y == 0).sum()), "fake": int((y == 1).sum())},
        "loss_unweighted": total_loss / count,
    }
    if args.split == "dev":
        sweep = [metrics(y, p, float(t)) for t in np.round(np.arange(0.30, 0.801, 0.01), 2)]
        best = max(sweep, key=lambda m: (m["balanced_accuracy"] if m["balanced_accuracy"] is not None else -1, m["macro_f1"]))
        threshold = args.threshold if args.threshold is not None else best["threshold"]
        result["threshold_selection"] = "user_supplied" if args.threshold is not None else "maximum_dev_balanced_accuracy"
        result["best_dev_threshold_metrics"] = best
        result["threshold_sweep"] = sweep
    else:
        if args.threshold is not None:
            threshold, source = args.threshold, "user_supplied"
        else:
            dev_file = ROOT / "results" / f"multimodal_{mode}_dev_metrics_full.json"
            if dev_file.exists():
                dev_result = json.loads(dev_file.read_text(encoding="utf-8"))
                threshold, source = float(dev_result["selected_threshold"]), "saved_dev_selected_threshold"
            else:
                threshold, source = 0.5, "fallback_0.5_no_dev_result_found"
                print("WARNING: dev metrics absent; using 0.5. Run dev evaluation first.")
        result["threshold_source"] = source
    result["selected_threshold"] = float(threshold)
    result["metrics_at_selected_threshold"] = metrics(y, p, threshold)

    final_path = ROOT / "results" / f"multimodal_{mode}_{args.split}_metrics_full.json"
    final_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nFULL {args.split.upper()} RESULT")
    print(json.dumps(result, indent=2))
    print(f"\nSaved metrics: {final_path}")
    print(f"Saved chunks: {out_dir}")


if __name__ == "__main__":
    main()
