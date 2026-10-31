import csv
import math
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.lavdf_dataset import (  # noqa: E402
    DATASET_ROOT,
    METADATA_CSV,
    LAVDFDataset,
)
from preprocessing.audio_extractor import extract_audio  # noqa: E402

SR = 16000
WINDOW = 5.0


def main():
    meta = {}
    with METADATA_CSV.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if (DATASET_ROOT / row["video_path"]).is_file():
                meta[row["sample_id"]] = row
    print(f"\nLocal videos found: {len(meta)}")

    print("\n=== 1. Audio extraction (first 6 videos) ===")
    problems = 0
    with tempfile.TemporaryDirectory() as tmp:
        for sample_id, row in list(meta.items())[:6]:
            wav = Path(tmp) / f"{sample_id}.wav"
            extract_audio(DATASET_ROOT / row["video_path"], wav)
            audio, sr = sf.read(str(wav))
            duration = len(audio) / sr
            expected = float(row["duration"])
            ok = sr == SR and audio.ndim == 1 and abs(duration - expected) < 0.5
            problems += 0 if ok else 1
            print(
                f"RESULT {sample_id}: sr={sr}, mono={audio.ndim == 1}, "
                f"audio={duration:.2f}s, metadata={expected:.2f}s -> "
                f"{'OK' if ok else 'PROBLEM'}"
            )

    print("\n=== 2. Window alignment ===")
    for split in ("train", "dev"):
        dataset = LAVDFDataset(split=split)
        groups = defaultdict(list)
        for s in dataset.samples:
            groups[s["sample_id"]].append(s)
        for sample_id, windows in groups.items():
            windows.sort(key=lambda w: w["start"])
            duration = float(meta[sample_id]["duration"])
            starts_ok = all(
                abs(w["start"] - i * WINDOW) < 1e-6 for i, w in enumerate(windows)
            )
            ends_ok = abs(windows[-1]["end"] - duration) < 0.05
            expected_n = math.ceil(duration / WINDOW)
            if not (starts_ok and ends_ok):
                problems += 1
            print(
                f"RESULT {split} {sample_id}: duration={duration:.2f}s, "
                f"windows={len(windows)} (plain split={expected_n}), "
                f"starts_ok={starts_ok}, ends_ok={ends_ok}"
            )
        first = dataset.samples[0]
        feature = dataset._load_audio(first["video_path"], first["start"], first["end"])
        print(f"RESULT {split}: log-mel shape of first window = {tuple(feature.shape)}")

    print(f"\nTotal problems: {problems}")


if __name__ == "__main__":
    main()