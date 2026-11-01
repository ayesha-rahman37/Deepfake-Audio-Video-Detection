"""
Generate final project plots from existing experiment results.

Run from project root:
    python evaluation/generate_final_plots.py

No model training or GPU inference is performed.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "figures"

OUT.mkdir(parents=True, exist_ok=True)


def read_json(filename):
    path = RESULTS / filename

    if not path.exists():
        print(f"[SKIP] Missing file: {path}")
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------
# 1. ROC CURVE
# --------------------------------------------------

def plot_roc():

    folder = RESULTS / "reliability_chunks_reliability"
    files = sorted(folder.glob("part_*.npz"))

    if not files:
        print("[SKIP] No saved prediction chunks for ROC.")
        return

    ys = []
    ps = []

    for file in files:

        with np.load(file) as data:

            y_key = next(
                (k for k in ("labels", "y", "targets") if k in data.files),
                None
            )

            p_key = next(
                (
                    k for k in (
                        "probabilities",
                        "probs",
                        "scores",
                        "predictions"
                    )
                    if k in data.files
                ),
                None
            )

            if y_key is None or p_key is None:
                print(
                    f"[SKIP] Cannot identify arrays in {file.name}: "
                    f"{data.files}"
                )
                return

            ys.append(
                np.asarray(data[y_key]).reshape(-1).astype(int)
            )

            ps.append(
                np.asarray(data[p_key]).reshape(-1).astype(float)
            )

    y = np.concatenate(ys)
    p = np.concatenate(ps)

    if len(y) != len(p) or len(np.unique(y)) != 2:
        print("[SKIP] Invalid binary labels or prediction scores.")
        return

    fpr, tpr, _ = roc_curve(y, p)
    roc_score = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        fpr,
        tpr,
        label=f"Reliability Fusion (AUC={roc_score:.3f})"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        label="Chance"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ax.set_title(
        "ROC Curve - Full LAV-DF Test Split"
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        OUT / "roc_reliability_full_test.png",
        dpi=220
    )

    plt.close(fig)

    print(
        f"[OK] ROC curve generated. "
        f"Samples={len(y)}, AUC={roc_score:.4f}"
    )


# --------------------------------------------------
# 2. CONFUSION MATRIX
# --------------------------------------------------

def plot_confusion():

    obj = read_json(
        "multimodal_reliability_test_metrics_full.json"
    )

    if not obj:
        return

    cm = obj.get("confusion_matrix_labels_0_1")

    if cm is None:

        for key in ("metrics", "results", "overall"):

            if isinstance(obj.get(key), dict):

                cm = obj[key].get(
                    "confusion_matrix_labels_0_1"
                )

                if cm is not None:
                    break

    if cm is None:
        print("[SKIP] Confusion matrix not found.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(cm),
        display_labels=["Real", "Fake"]
    ).plot(
        ax=ax,
        values_format="d",
        colorbar=False
    )

    ax.set_title(
        "Confusion Matrix - Reliability Fusion (Full Test)"
    )

    fig.tight_layout()

    fig.savefig(
        OUT / "confusion_matrix_reliability_full_test.png",
        dpi=220
    )

    plt.close(fig)

    print("[OK] Confusion matrix generated.")


# --------------------------------------------------
# 3. FUSION COMPARISON AND ABLATION
# --------------------------------------------------

def plot_ablation():

    obj = read_json(
        "model_comparison_ablation_test_256.json"
    )

    if not obj:
        return

    rows = []

    models = obj.get("models", {})

    for model_name, model_data in models.items():

        variants = model_data.get("results", {})

        for variant_name, metrics in variants.items():

            if (
                isinstance(metrics, dict)
                and "accuracy" in metrics
            ):

                rows.append(
                    (f"{model_name}_{variant_name}", metrics)
                )

    if not rows:
        print("[SKIP] No ablation metric rows found.")
        return

    labels = [
        name.replace("_", "\n")
        for name, _ in rows
    ]

    x = np.arange(len(rows))
    width = 0.36

    accuracy = [
        metrics.get("accuracy", np.nan)
        for _, metrics in rows
    ]

    macro_f1 = [
        metrics.get("macro_f1", np.nan)
        for _, metrics in rows
    ]

    fig, ax = plt.subplots(
        figsize=(max(9, len(rows) * 1.25), 5.5)
    )

    ax.bar(
        x - width / 2,
        accuracy,
        width,
        label="Accuracy"
    )

    ax.bar(
        x + width / 2,
        macro_f1,
        width,
        label="Macro-F1"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=20,
        ha="right"
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")

    ax.set_title(
        "Fusion Comparison and Modality Ablation "
        "(256-sample subset)"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        OUT / "fusion_ablation_comparison.png",
        dpi=220
    )

    plt.close(fig)

    print("[OK] Fusion comparison and ablation generated.")


# --------------------------------------------------
# 4. ROBUSTNESS COMPARISON
# --------------------------------------------------

def plot_robustness():

    obj = read_json(
        "robustness_benchmark_reliability_test.json"
    )

    if not obj:
        return

    data = obj.get("results", obj)

    if not isinstance(data, dict):
        print("[SKIP] Unexpected robustness JSON structure.")
        return

    rows = [
        (name, metrics)
        for name, metrics in data.items()
        if (
            isinstance(metrics, dict)
            and "accuracy" in metrics
        )
    ]

    if not rows:
        print("[SKIP] No robustness metrics found.")
        return

    labels = [
        name.replace("_", "\n")
        for name, _ in rows
    ]

    x = np.arange(len(rows))
    width = 0.36

    accuracy = [
        metrics.get("accuracy", np.nan)
        for _, metrics in rows
    ]

    roc_auc = [
        metrics.get("roc_auc", np.nan)
        for _, metrics in rows
    ]

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.bar(
        x - width / 2,
        accuracy,
        width,
        label="Accuracy"
    )

    ax.bar(
        x + width / 2,
        roc_auc,
        width,
        label="ROC-AUC"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=20,
        ha="right"
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")

    ax.set_title(
        "Preliminary Robustness Benchmark "
        "(256-sample subset)"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        OUT / "robustness_comparison.png",
        dpi=220
    )

    plt.close(fig)

    print("[OK] Robustness comparison generated.")


# --------------------------------------------------
# 5. TRADITIONAL ML COMPARISON
# --------------------------------------------------

def plot_traditional_ml():

    obj = read_json(
        "traditional_ml_baseline.json"
    )

    if not obj:
        return

    rows = obj.get("results", [])

    if isinstance(rows, dict):

        rows = [
            {
                "model": name,
                **metrics
            }
            for name, metrics in rows.items()
            if isinstance(metrics, dict)
        ]

    if not isinstance(rows, list):
        print("[SKIP] Unexpected traditional ML format.")
        return

    rows = [
        row
        for row in rows
        if (
            isinstance(row, dict)
            and "accuracy" in row
        )
    ]

    if not rows:
        print("[SKIP] No SVM/Random Forest results found.")
        return

    labels = [
        row.get("model", f"Model {i + 1}").replace("_", "\n")
        for i, row in enumerate(rows)
    ]

    metric_list = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("roc_auc", "ROC-AUC")
    ]

    x = np.arange(len(rows))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (metric_key, metric_label) in enumerate(metric_list):

        values = [
            row.get(metric_key, np.nan)
            for row in rows
        ]

        ax.bar(
            x + (i - 1) * width,
            values,
            width,
            label=metric_label
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")

    ax.set_title(
        "Traditional ML Baseline "
        "(2,048 Train / 512 Test Subset)"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        OUT / "traditional_ml_comparison.png",
        dpi=220
    )

    plt.close(fig)

    print("[OK] Traditional ML comparison generated.")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    print("\n===== GENERATING FINAL PROJECT FIGURES =====\n")

    plot_roc()

    plot_confusion()

    plot_ablation()

    plot_robustness()

    plot_traditional_ml()

    print("\n===== FINISHED =====")

    print(f"\nFigures saved in: {OUT}")