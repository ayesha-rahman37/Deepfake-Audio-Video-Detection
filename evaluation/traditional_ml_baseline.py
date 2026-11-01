"""Classical ML baseline on embeddings from the trained multimodal detector.

Extracts frozen video/audio embeddings from a trained multimodal checkpoint,
then trains SVM and Random Forest classifiers. Uses window-level LAV-DF labels.
This is an embedding-based traditional ML baseline, not a raw-pixel baseline.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.lavdf_dataset import LAVDFDataset
from models.multimodal_model import MultimodalDeepfakeDetector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/multimodal_reliability_best.pt")
    p.add_argument("--train-samples", type=int, default=2048)
    p.add_argument("--test-samples", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fusion-mode", choices=["reliability", "concat"], default="reliability")
    p.add_argument("--output", default="results/traditional_ml_baseline.json")
    return p.parse_args()


def choose_balanced_indices(dataset, limit, seed):
    labels = np.asarray([int(x["label"]) for x in dataset.samples], dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = []
    for cls in (0, 1):
        candidates = np.flatnonzero(labels == cls)
        if len(candidates) == 0:
            continue
        take = min(len(candidates), max(1, limit // 2))
        selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
    rng.shuffle(selected)
    return selected


def load_checkpoint(model, checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(payload, dict):
        state = payload.get("model_state_dict", payload.get("state_dict", payload))
    else:
        state = payload
    # Handle common DataParallel prefixes without changing tensor values.
    if state and all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


@torch.inference_mode()
def extract_features(dataset, indices, model, device, batch_size):
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    features, labels = [], []
    started = time.time()
    for step, batch in enumerate(loader, 1):
        video = batch["video"].to(device, non_blocking=True)
        audio = batch["audio"].to(device, non_blocking=True)
        output = model(video, audio)
        # Concatenate modality embeddings and quality proxies. The feature
        # extractor is fixed; only the classical classifier is fitted.
        vector = torch.cat(
            [output["video_embedding"], output["audio_embedding"],
             output["quality_features"]], dim=1
        )
        features.append(vector.detach().cpu().numpy().astype(np.float32))
        labels.extend(batch["label"].cpu().numpy().astype(int).tolist())
        if step % 100 == 0 or step == len(loader):
            print(f"  extracted {step}/{len(loader)} batches")
    x = np.concatenate(features, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    print(f"  feature extraction: {len(y)} samples in {time.time()-started:.1f}s; "
          f"feature_dim={x.shape[1]}")
    return x, y


def evaluate_classifier(name, clf, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    score = clf.predict_proba(x_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(x_test)
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "precision_fake": float(precision_score(y_test, pred, zero_division=0)),
        "recall_fake": float(recall_score(y_test, pred, zero_division=0)),
        "f1_fake": float(f1_score(y_test, pred, zero_division=0)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, score)) if len(np.unique(y_test)) == 2 else None,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, pred, labels=[0, 1], target_names=["real", "fake"],
            output_dict=True, zero_division=0
        ),
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    train_ds = LAVDFDataset("train")
    test_ds = LAVDFDataset("test")
    train_idx = choose_balanced_indices(train_ds, args.train_samples, args.seed)
    test_idx = choose_balanced_indices(test_ds, args.test_samples, args.seed + 1)
    print(f"Selected train={len(train_idx)}, test={len(test_idx)} windows")

    model = MultimodalDeepfakeDetector(
        pretrained_visual=False, freeze_visual=True,
        frame_batch_size=4, fusion_mode=args.fusion_mode
    ).to(device).eval()
    load_checkpoint(model, checkpoint, device)

    print("Extracting train embeddings...")
    x_train, y_train = extract_features(train_ds, train_idx, model, device, args.batch_size)
    print("Extracting test embeddings...")
    x_test, y_test = extract_features(test_ds, test_idx, model, device, args.batch_size)

    classifiers = [
        ("SVM_RBF", make_pipeline(
            StandardScaler(), SVC(C=1.0, kernel="rbf", probability=True,
                                  class_weight="balanced", random_state=args.seed)
        )),
        ("Random_Forest", RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=args.seed, n_jobs=-1
        )),
    ]
    results = []
    for name, clf in classifiers:
        print(f"Training/evaluating {name}...")
        results.append(evaluate_classifier(name, clf, x_train, y_train, x_test, y_test))

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "traditional_ml_on_frozen_multimodal_embeddings",
        "feature_definition": "512-d video embedding + 256-d audio embedding + 4 quality indicators",
        "fusion_mode_used_for_feature_extraction": args.fusion_mode,
        "checkpoint": str(checkpoint),
        "train_split": "train",
        "test_split": "test",
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "train_class_counts": {"real": int((y_train == 0).sum()), "fake": int((y_train == 1).sum())},
        "test_class_counts": {"real": int((y_test == 0).sum()), "fake": int((y_test == 1).sum())},
        "feature_dim": int(x_train.shape[1]),
        "seed": args.seed,
        "results": results,
        "caveat": "Subset-based embedding baseline. Compare only on the same fixed test indices "
                  "as other models; this is not a full-test result or raw-feature classical baseline."
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {output}")
    for item in results:
        print(f"{item['model']}: accuracy={item['accuracy']:.4f}, "
              f"macro_f1={item['macro_f1']:.4f}, auc={item['roc_auc']}")


if __name__ == "__main__":
    main()
