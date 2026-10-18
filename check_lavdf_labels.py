
from collections import Counter
from datasets.lavdf_dataset import LAVDFDataset

for split in ("train", "dev", "test"):
    dataset = LAVDFDataset(split=split)
    counts = Counter(sample["label"] for sample in dataset.samples)

    print(
        f"{split}: total={len(dataset)}, "
        f"real={counts[0]}, fake={counts[1]}"
    )
