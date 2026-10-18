
import csv
import json
from pathlib import Path


# ===== PROJECT AND DATASET PATHS =====
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(r"D:\LAV-DF")

INPUT_METADATA = DATASET_ROOT / "metadata.min.json"
OUTPUT_METADATA = (
    PROJECT_ROOT / "data" / "metadata" / "lavdf_metadata.csv"
)


def main():
    if not INPUT_METADATA.is_file():
        raise FileNotFoundError(
            f"LAV-DF metadata not found: {INPUT_METADATA}"
        )

    print("Loading LAV-DF metadata...")

    with INPUT_METADATA.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("Expected metadata to contain a JSON list.")

    OUTPUT_METADATA.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "sample_id",
        "video_path",
        "split",
        "label",
        "n_fakes",
        "fake_periods",
        "duration",
        "original",
        "modify_video",
        "modify_audio",
        "video_frames",
        "audio_channels",
        "audio_frames",
    ]

    missing_files = []
    rows_written = 0
    label_counts = {"real": 0, "fake": 0}
    split_counts = {}

    with OUTPUT_METADATA.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for record in records:
            relative_path = record.get("file")

            if not relative_path:
                missing_files.append("(empty metadata path)")
                continue

            video_path = DATASET_ROOT / relative_path

            if not video_path.is_file():
                missing_files.append(relative_path)
                continue

            n_fakes = int(record.get("n_fakes", 0) or 0)

            # Clip-level label:
            # fake means at least one annotated fake segment.
            label = "fake" if n_fakes > 0 else "real"

            split = record.get("split", "unknown")

            row = {
                "sample_id": Path(relative_path).stem,
                # Keep paths relative to D:\LAV-DF.
                "video_path": relative_path.replace("\\", "/"),
                "split": split,
                "label": label,
                "n_fakes": n_fakes,
                "fake_periods": json.dumps(
                    record.get("fake_periods", []),
                    separators=(",", ":"),
                ),
                "duration": record.get("duration"),
                "original": record.get("original"),
                "modify_video": record.get("modify_video"),
                "modify_audio": record.get("modify_audio"),
                "video_frames": record.get("video_frames"),
                "audio_channels": record.get("audio_channels"),
                "audio_frames": record.get("audio_frames"),
            }

            writer.writerow(row)
            rows_written += 1

            label_counts[label] += 1
            split_counts[split] = split_counts.get(split, 0) + 1

    print("\n===== LAV-DF METADATA REPORT =====")
    print(f"Input records: {len(records)}")
    print(f"CSV rows written: {rows_written}")
    print(f"Missing video files: {len(missing_files)}")
    print(f"Real clips: {label_counts['real']}")
    print(f"Fake clips: {label_counts['fake']}")

    print("\nSplit counts:")
    for split, count in sorted(split_counts.items()):
        print(f"  {split}: {count}")

    print(f"\nOutput CSV: {OUTPUT_METADATA}")

    if missing_files:
        print("\nFirst 10 missing paths:")
        for path in missing_files[:10]:
            print(f"  {path}")

        print("\nWARNING: Some records were skipped.")
    elif rows_written == len(records):
        print("\nPASS: All metadata records were converted.")
    else:
        print("\nWARNING: CSV row count differs from input count.")


if __name__ == "__main__":
    main()
