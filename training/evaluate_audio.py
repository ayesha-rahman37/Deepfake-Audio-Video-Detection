import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.audio_encoder import AudioEncoder  # noqa: E402
from training.train_audio import (  # noqa: E402
    CHECKPOINT_PATH,
    DummyAudioDataset,
    build_real_dataset_class,
    normalize,
)
from utils.device import device  # noqa: E402

REPORT_PATH = PROJECT_ROOT / "results" / "audio_report.json"


def auc_score(labels, scores):
    pos = [s for l, s in zip(labels, scores) if l == 1]
    neg = [s for l, s in zip(labels, scores) if l == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def safe_div(a, b):
    return a / b if b else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"No checkpoint at {CHECKPOINT_PATH}. Run train_audio.py first."
        )

    if args.dummy:
        dataset = DummyAudioDataset(32)
        samples = None
    else:
        dataset = build_real_dataset_class()(args.split)
        samples = dataset.samples

    indices = list(range(len(dataset)))
    if args.max_samples and len(indices) > args.max_samples:
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(dataset), generator=generator)[
            : args.max_samples
        ].tolist()

    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size)

    model = AudioEncoder().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    probs, labels = [], []
    with torch.no_grad():
        for audio, label in loader:
            _, logit = model(normalize(audio).to(device))
            probs += torch.sigmoid(logit).cpu().tolist()
            labels += [int(x) for x in label.tolist()]

    preds = [1 if p > 0.5 else 0 for p in probs]
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    wrong = []
    for position, (p, prob, l) in enumerate(zip(preds, probs, labels)):
        if p != l:
            info = {"true": l, "predicted": p, "fake_probability": round(prob, 3)}
            if samples is not None:
                s = samples[indices[position]]
                info.update(
                    sample_id=s["sample_id"],
                    start=s["start"],
                    end=s["end"],
                )
            wrong.append(info)
    wrong.sort(key=lambda w: abs(w["fake_probability"] - w["true"]), reverse=True)

    report = {
        "split": "dummy" if args.dummy else args.split,
        "windows_evaluated": len(labels),
        "accuracy": round(safe_div(tp + tn, len(labels)), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc": auc_score(labels, probs),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "worst_wrong_predictions": wrong[:10],
    }

    print(json.dumps(report, indent=2))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()