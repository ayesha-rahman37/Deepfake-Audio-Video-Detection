
import json
from pathlib import Path
from collections import Counter

# Local LAV-DF dataset
DATASET_DIR = Path(r"D:\LAV-DF")
METADATA_FILE = DATASET_DIR / "metadata.min.json"


def main():
    if not METADATA_FILE.exists():
        print(f"ERROR: Metadata file not found:\n{METADATA_FILE}")
        return

    print("Loading LAV-DF metadata...")
    with METADATA_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    print(f"\nTotal metadata records: {len(records)}")

    split_counts = Counter()
    label_counts = Counter()
    video_modality_counts = Counter()
    audio_modality_counts = Counter()

    missing_files = []
    existing_files = 0
    invalid_paths = 0

    for record in records:
        relative_path = record.get("file")

        split_counts[record.get("split", "unknown")] += 1

        if record.get("n_fakes", 0) > 0:
            label_counts["fake"] += 1
        else:
            label_counts["real"] += 1

        video_modality_counts[bool(record.get("modify_video", False))] += 1
        audio_modality_counts[bool(record.get("modify_audio", False))] += 1

        if not relative_path:
            invalid_paths += 1
            continue

        video_path = DATASET_DIR / relative_path

        if video_path.is_file():
            existing_files += 1
        else:
            missing_files.append(relative_path)

    print("\n--- SPLIT COUNTS ---")
    for split, count in sorted(split_counts.items()):
        print(f"{split}: {count}")

    print("\n--- LABEL COUNTS ---")
    print(f"Real records (n_fakes = 0): {label_counts['real']}")
    print(f"Fake records (n_fakes > 0): {label_counts['fake']}")

    print("\n--- MODALITY COUNTS ---")
    print(f"Video modified: {video_modality_counts[True]}")
    print(f"Audio modified: {audio_modality_counts[True]}")

    print("\n--- FILE CHECK ---")
    print(f"Existing video files: {existing_files}")
    print(f"Missing video files: {len(missing_files)}")
    print(f"Invalid/missing paths: {invalid_paths}")

    if missing_files:
        print("\nFirst 10 missing paths:")
        for path in missing_files[:10]:
            print(f"  {path}")

    if not missing_files and invalid_paths == 0:
        print("\nPASS: All metadata paths point to existing files.")
    else:
        print("\nCHECK REQUIRED: Some metadata paths need investigation.")


if __name__ == "__main__":
    main()
