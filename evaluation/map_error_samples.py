import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

CHUNK_DIR = ROOT / "results" / "reliability_chunks_reliability"
ERROR_CSV = ROOT / "results" / "error_analysis" / "reliability_test_errors.csv"
OUTPUT_CSV = ROOT / "results" / "error_analysis" / "reliability_mapped_errors.csv"

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.lavdf_dataset import LAVDFDataset, DATASET_ROOT


def main():
    print("Loading test dataset metadata only...")
    dataset = LAVDFDataset(split="test")

    total = len(dataset)
    manifest_path = CHUNK_DIR / "manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if total != int(manifest["dataset_samples"]):
        raise RuntimeError(
            f"Dataset has {total} samples but chunks expect "
            f"{manifest['dataset_samples']}. Stop: ordering may differ."
        )

    parts = int(manifest["parts"])
    boundaries = np.linspace(0, total, parts + 1, dtype=int)

    all_labels = []
    all_probs = []

    print(f"Checking {parts} saved chunks...")

    for part in range(parts):
        path = CHUNK_DIR / f"part_{part + 1:02d}.npz"

        if not path.is_file():
            raise FileNotFoundError(f"Missing chunk: {path.name}")

        with np.load(path) as data:
            expected_start = int(boundaries[part])
            expected_end = int(boundaries[part + 1])

            if int(data["start"]) != expected_start:
                raise RuntimeError(f"Wrong start index in {path.name}")

            if int(data["end"]) != expected_end:
                raise RuntimeError(f"Wrong end index in {path.name}")

            labels = data["labels"].copy()
            probs = data["probabilities"].copy()

            if len(labels) != expected_end - expected_start:
                raise RuntimeError(f"Wrong label count in {path.name}")

            if len(probs) != len(labels):
                raise RuntimeError(f"Prediction count mismatch in {path.name}")

            all_labels.extend(labels.tolist())
            all_probs.extend(probs.tolist())

    if len(all_labels) != total:
        raise RuntimeError("Combined chunk count does not match dataset.")

    print("Verifying original dataset label order...")

    for index, sample in enumerate(dataset.samples):
        if int(sample["label"]) != int(all_labels[index]):
            raise RuntimeError(
                f"Label/order mismatch at sample index {index}. "
                "Do not create mapped CSV."
            )

    print("Dataset ordering and labels verified.")

    probabilities = np.asarray(all_probs)
    labels = np.asarray(all_labels)
    predictions = (probabilities >= 0.5).astype(np.int8)

    rows = []

    with ERROR_CSV.open("r", encoding="utf-8-sig", newline="") as file:
        errors = csv.DictReader(file)

        for error in errors:
            index = int(error["sample_index"])

            if index < 0 or index >= total:
                raise IndexError(f"Invalid sample index: {index}")

            sample = dataset.samples[index]

            if int(error["actual_label"]) != int(labels[index]):
                raise RuntimeError(f"Actual-label mismatch at index {index}")

            if int(error["predicted_label"]) != int(predictions[index]):
                raise RuntimeError(f"Prediction mismatch at index {index}")

            video_path = Path(sample["video_path"])

            try:
                relative_video = video_path.relative_to(DATASET_ROOT)
            except ValueError:
                relative_video = video_path

            rows.append({
                "sample_index": index,
                "sample_id": sample["sample_id"],
                "video_path": str(relative_video),
                "start_seconds": round(float(sample["start"]), 4),
                "end_seconds": round(float(sample["end"]), 4),
                "actual_label": int(labels[index]),
                "predicted_label": int(predictions[index]),
                "fake_probability": float(probabilities[index]),
                "error_type": error["error_type"],
            })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_index",
        "sample_id",
        "video_path",
        "start_seconds",
        "end_seconds",
        "actual_label",
        "predicted_label",
        "fake_probability",
        "error_type",
    ]

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nSUCCESS")
    print(f"Total test windows: {total}")
    print(f"Mapped error samples: {len(rows)}")
    print(f"Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()