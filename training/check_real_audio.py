"""Validate LAV-DF audio extraction and temporal window alignment.

Run from the project root:
    python training/check_real_audio.py

This script only reads the dataset and writes temporary WAV files. It does not
train, run inference, modify dataset files, or touch evaluation chunks.
"""

import csv
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.lavdf_dataset import (  # noqa: E402
    DATASET_ROOT,
    METADATA_CSV,
    LAVDFDataset,
    _read_video_info,
)
from preprocessing.audio_extractor import extract_audio  # noqa: E402

EXPECTED_SAMPLE_RATE = 16000
WINDOW_SECONDS = 5.0
AUDIO_SAMPLE_COUNT = 6
TOLERANCE = 0.05


def load_metadata():
    if not METADATA_CSV.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")

    rows = {}
    with METADATA_CSV.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            sample_id = (row.get("sample_id") or "").strip()
            relative_path = (row.get("video_path") or "").strip()
            if not sample_id or not relative_path:
                continue

            path = (DATASET_ROOT / relative_path).resolve()
            if path.is_file():
                row["_path"] = path
                rows[sample_id] = row
    return rows


def check_audio(metadata):
    print("\n=== AUDIO EXTRACTION: first 6 local videos ===")
    checked = failed = 0

    with tempfile.TemporaryDirectory(prefix="lavdf_audio_check_") as temp:
        for sample_id, row in list(metadata.items())[:AUDIO_SAMPLE_COUNT]:
            checked += 1
            wav_path = Path(temp) / f"{sample_id}.wav"
            try:
                extract_audio(row["_path"], wav_path)
                if not wav_path.is_file():
                    raise RuntimeError("Audio extractor did not create a WAV file")

                audio, sr = sf.read(str(wav_path))
                duration = len(audio) / sr if sr else 0.0
                expected = float(row["duration"])
                mono = audio.ndim == 1
                finite = bool(np.isfinite(audio).all())

                # Allow small container/codec duration differences.
                ok = (
                    sr == EXPECTED_SAMPLE_RATE
                    and mono
                    and finite
                    and abs(duration - expected) <= 0.5
                )
                if not ok:
                    failed += 1

                print(
                    f"{'PASS' if ok else 'FAIL'} {sample_id}: "
                    f"sr={sr}, mono={mono}, audio={duration:.3f}s, "
                    f"metadata={expected:.3f}s, finite={finite}"
                )
            except Exception as exc:
                failed += 1
                print(f"FAIL {sample_id}: {type(exc).__name__}: {exc}")

    print(f"Audio summary: {checked - failed}/{checked} passed")
    return checked, failed


def expected_dataset_end(metadata_duration, actual_duration, fps):
    """Mirror the dataset loader's short-final-window duration correction."""
    remainder = metadata_duration % WINDOW_SECONDS
    expected = min(metadata_duration, actual_duration)

    if 1e-6 < remainder <= 0.5:
        final_start = metadata_duration - remainder
        one_frame = 1.0 / fps
        if actual_duration <= final_start + one_frame:
            expected = min(actual_duration, final_start)

    return expected


def check_split(split, metadata):
    print(f"\n=== {split.upper()} WINDOW ALIGNMENT ===")
    dataset = LAVDFDataset(split=split)

    grouped = defaultdict(list)
    for sample in dataset.samples:
        grouped[sample["sample_id"]].append(sample)

    checked = failed = duration_adjustments = 0

    for sample_id, windows in grouped.items():
        checked += 1
        windows.sort(key=lambda item: item["start"])
        row = metadata.get(sample_id)

        if row is None:
            failed += 1
            print(f"FAIL {sample_id}: metadata row not found")
            continue

        metadata_duration = float(row["duration"])
        starts_ok = abs(float(windows[0]["start"])) <= TOLERANCE

        lengths_ok = all(
            0 < float(item["end"]) - float(item["start"])
            <= WINDOW_SECONDS + 1e-6
            for item in windows
        )
        contiguous_ok = all(
            abs(float(windows[i]["start"]) - float(windows[i - 1]["end"]))
            <= TOLERANCE
            for i in range(1, len(windows))
        )

        last_end = float(windows[-1]["end"])
        boundary_ok = abs(last_end - metadata_duration) <= TOLERANCE
        actual_duration = None
        expected_end = metadata_duration

        # Most samples match metadata directly. Open the video container only
        # for mismatches, where the dataset loader may have corrected a short tail.
        if not boundary_ok:
            actual_duration, fps, _ = _read_video_info(row["_path"])
            expected_end = expected_dataset_end(
                metadata_duration, actual_duration, fps
            )
            boundary_ok = (
                abs(last_end - expected_end) <= max(TOLERANCE, 1.0 / fps)
            )
            if boundary_ok:
                duration_adjustments += 1

        ok = starts_ok and lengths_ok and contiguous_ok and boundary_ok
        if not ok:
            failed += 1

        print(
            f"{'PASS' if ok else 'FAIL'} {sample_id}: "
            f"windows={len(windows)}, last_end={last_end:.3f}s, "
            f"metadata={metadata_duration:.3f}s, "
            f"actual={actual_duration if actual_duration is not None else 'not-probed'}, "
            f"expected_end={expected_end:.3f}s, "
            f"starts={starts_ok}, lengths={lengths_ok}, "
            f"contiguous={contiguous_ok}, boundary={boundary_ok}"
        )

    print(
        f"{split.upper()} summary: {checked - failed}/{checked} passed; "
        f"duration adjustments={duration_adjustments}; failures={failed}"
    )

    # Check the dataset's actual training-time Log-Mel path on one sample.
    try:
        sample = dataset.samples[0]
        feature = dataset._load_audio(
            sample["video_path"], sample["start"], sample["end"]
        )
        feature_array = feature.detach().cpu().numpy()
        feature_ok = (
            feature_array.ndim == 2
            and feature_array.shape[0] == 64
            and np.isfinite(feature_array).all()
        )
        print(
            f"{'PASS' if feature_ok else 'FAIL'} {split} Log-Mel: "
            f"shape={tuple(feature_array.shape)}, finite={bool(np.isfinite(feature_array).all())}"
        )
        if not feature_ok:
            failed += 1
    except Exception as exc:
        failed += 1
        print(f"FAIL {split} Log-Mel: {type(exc).__name__}: {exc}")

    return checked, failed


def main():
    print("LAV-DF audio and temporal alignment validation")
    print(f"Dataset root: {DATASET_ROOT}")

    metadata = load_metadata()
    print(f"Local videos found: {len(metadata)}")
    if not metadata:
        raise RuntimeError(
            "No local dataset videos found. Check LAVDF_ROOT and metadata paths."
        )

    audio_checked, audio_failed = check_audio(metadata)

    total_groups = total_failures = 0
    for split in ("train", "dev"):
        groups, failures = check_split(split, metadata)
        total_groups += groups
        total_failures += failures

    total_failures += audio_failed
    print("\n=== FINAL SUMMARY ===")
    print(f"Audio files checked: {audio_checked}")
    print(f"Video sample groups checked: {total_groups}")
    print(f"Validation failures: {total_failures}")

    if total_failures:
        print("RESULT: CHECK FAILED — inspect the FAIL lines.")
        raise SystemExit(1)

    print("RESULT: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
