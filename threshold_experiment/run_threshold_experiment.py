
"""
Development-Set-Constrained Operating Threshold Optimization
for False-Positive Reduction in Multimodal Deepfake Detection.

This script:
1. Loads existing development and test prediction chunks.
2. Selects a decision threshold using development predictions only.
3. Keeps development fake recall >= 90%.
4. Evaluates the selected threshold on the held-out test set.
5. Compares the optimized threshold with the default 0.50 threshold.
6. Saves metrics, CSV files, and plots.

IMPORTANT:
- No model retraining is performed.
- Model weights and architecture remain unchanged.
- Test predictions are not used for threshold selection.
"""

from pathlib import Path
import json
import csv

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEV_DIR = RESULTS_DIR / "reliability_chunks_dev_reliability"

TEST_DIR = RESULTS_DIR / "reliability_chunks_reliability"

MIN_DEV_FAKE_RECALL = 0.90

DEFAULT_THRESHOLD = 0.50

SWEEP_MIN = 0.30
SWEEP_MAX = 0.70
SWEEP_STEP = 0.005


# ============================================================
# 2. CHECK REQUIRED DIRECTORIES
# ============================================================

print("\n" + "=" * 65)
print("MULTIMODAL DEEPFAKE THRESHOLD OPTIMIZATION")
print("=" * 65)

print("\nProject directory:")
print(PROJECT_ROOT)

print("\nDevelopment directory:")
print(DEV_DIR)

print("\nTest directory:")
print(TEST_DIR)

if not RESULTS_DIR.exists():
    raise FileNotFoundError(
        f"Results directory does not exist: {RESULTS_DIR}"
    )

if not DEV_DIR.exists():
    raise FileNotFoundError(
        f"Development prediction folder not found: {DEV_DIR}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test prediction folder not found: {TEST_DIR}"
    )


# ============================================================
# 3. LOAD EXISTING PREDICTION CHUNKS
# ============================================================

def load_chunks(folder):
    """
    Load all part_*.npz prediction files from a directory.

    Expected arrays in each NPZ:
        labels
        probabilities
    """

    files = sorted(folder.glob("part_*.npz"))

    if not files:
        raise FileNotFoundError(
            f"No part_*.npz files found inside: {folder}"
        )

    all_labels = []
    all_scores = []

    for file_path in files:

        print(f"Loading: {file_path.name}")

        with np.load(file_path) as data:

            if "labels" not in data:
                raise KeyError(
                    f"'labels' array missing in {file_path}"
                )

            if "probabilities" not in data:
                raise KeyError(
                    f"'probabilities' array missing in {file_path}"
                )

            labels = data["labels"].astype(np.int8)

            scores = data["probabilities"].astype(np.float64)

            if len(labels) != len(scores):
                raise ValueError(
                    f"Labels/scores length mismatch in {file_path}"
                )

            if not np.all(np.isfinite(scores)):
                raise ValueError(
                    f"Non-finite prediction scores in {file_path}"
                )

            if np.any((scores < 0) | (scores > 1)):
                raise ValueError(
                    f"Prediction scores outside [0, 1] in {file_path}"
                )

            all_labels.append(labels)
            all_scores.append(scores)

    labels = np.concatenate(all_labels)
    scores = np.concatenate(all_scores)

    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError(
            f"Unexpected labels in {folder}. Expected only 0 and 1."
        )

    return labels, scores, len(files)


print("\n" + "-" * 65)
print("LOADING DEVELOPMENT PREDICTIONS")
print("-" * 65)

y_dev, p_dev, dev_parts = load_chunks(DEV_DIR)

print("\n" + "-" * 65)
print("LOADING TEST PREDICTIONS")
print("-" * 65)

y_test, p_test, test_parts = load_chunks(TEST_DIR)


# ============================================================
# 4. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 65)
print("DATASET SUMMARY")
print("=" * 65)

print(f"Development chunks : {dev_parts}")
print(f"Development samples: {len(y_dev):,}")
print(f"Development real   : {np.sum(y_dev == 0):,}")
print(f"Development fake   : {np.sum(y_dev == 1):,}")

print()

print(f"Test chunks        : {test_parts}")
print(f"Test samples       : {len(y_test):,}")
print(f"Test real          : {np.sum(y_test == 0):,}")
print(f"Test fake          : {np.sum(y_test == 1):,}")


# ============================================================
# 5. METRIC CALCULATION FUNCTION
# ============================================================

def calculate_metrics(labels, scores, threshold):
    """
    Calculate binary classification metrics.

    Label convention:
        0 = Real
        1 = Fake

    Prediction rule:
        score >= threshold -> Fake
        score < threshold  -> Real
    """

    predictions = (scores >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),

        "accuracy": float(
            accuracy_score(labels, predictions)
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),

        "fake_precision": float(
            precision_score(
                labels,
                predictions,
                pos_label=1,
                zero_division=0
            )
        ),

        "fake_recall": float(
            recall_score(
                labels,
                predictions,
                pos_label=1,
                zero_division=0
            )
        ),

        "fake_f1": float(
            f1_score(
                labels,
                predictions,
                pos_label=1,
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(labels, scores)
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),

        "samples": int(len(labels)),
    }


# ============================================================
# 6. SELECT OPTIMAL THRESHOLD USING DEV SET ONLY
# ============================================================

print("\n" + "=" * 65)
print("DEVELOPMENT-SET THRESHOLD SELECTION")
print("=" * 65)

print(
    f"\nConstraint: Development fake recall >= "
    f"{MIN_DEV_FAKE_RECALL:.2f}"
)

print(
    "Objective: Maximize development fake precision"
)

# Sort predictions from highest score to lowest score.
order = np.argsort(-p_dev, kind="mergesort")

sorted_scores = p_dev[order]
sorted_labels = y_dev[order]

# Cumulative true positives and false positives.
cum_tp = np.cumsum(sorted_labels == 1)
cum_fp = np.cumsum(sorted_labels == 0)

# Last index of each unique score.
last_at_score = np.r_[
    np.where(sorted_scores[:-1] != sorted_scores[1:])[0],
    len(sorted_scores) - 1
]

total_fake = int(np.sum(y_dev == 1))

if total_fake == 0:
    raise ValueError(
        "Development dataset contains no fake samples."
    )

tp_values = cum_tp[last_at_score].astype(np.float64)

fp_values = cum_fp[last_at_score].astype(np.float64)

dev_recalls = tp_values / total_fake

dev_precisions = (
    tp_values / np.maximum(tp_values + fp_values, 1)
)

# Keep thresholds satisfying the recall constraint.
valid_indices = np.where(
    dev_recalls >= MIN_DEV_FAKE_RECALL
)[0]

if len(valid_indices) == 0:
    raise RuntimeError(
        "No threshold satisfies the development recall constraint."
    )

# Maximize precision.
# Tie-break by higher recall, then higher threshold.
best_index = max(
    valid_indices,
    key=lambda i: (
        dev_precisions[i],
        dev_recalls[i],
        sorted_scores[last_at_score[i]]
    )
)

selected_threshold = float(
    sorted_scores[last_at_score[best_index]]
)

best_dev_metrics = calculate_metrics(
    y_dev,
    p_dev,
    selected_threshold
)

print("\nSelected threshold:", selected_threshold)

print(
    "Development fake precision:",
    f"{best_dev_metrics['fake_precision'] * 100:.2f}%"
)

print(
    "Development fake recall:",
    f"{best_dev_metrics['fake_recall'] * 100:.2f}%"
)


# ============================================================
# 7. TEST SET EVALUATION
# ============================================================

print("\n" + "=" * 65)
print("HELD-OUT TEST SET EVALUATION")
print("=" * 65)

# Original operating threshold.
test_default_metrics = calculate_metrics(
    y_test,
    p_test,
    DEFAULT_THRESHOLD
)

# Development-selected threshold.
test_optimized_metrics = calculate_metrics(
    y_test,
    p_test,
    selected_threshold
)

default_fp = test_default_metrics["fp"]
optimized_fp = test_optimized_metrics["fp"]

if default_fp > 0:
    fp_reduction = (
        (default_fp - optimized_fp) / default_fp
    )
else:
    fp_reduction = 0.0


# ============================================================
# 8. PRINT COMPARISON
# ============================================================

print("\n" + "=" * 65)
print("TEST RESULT COMPARISON")
print("=" * 65)

print(
    f"\n{'Metric':<25}"
    f"{'Default 0.50':>18}"
    f"{'Optimized':>18}"
)

print("-" * 61)

comparison_metrics = [
    ("Accuracy", "accuracy"),
    ("Balanced Accuracy", "balanced_accuracy"),
    ("Fake Precision", "fake_precision"),
    ("Fake Recall", "fake_recall"),
    ("Fake F1", "fake_f1"),
    ("ROC-AUC", "roc_auc"),
]

for label, key in comparison_metrics:

    default_value = (
        test_default_metrics[key] * 100
    )

    optimized_value = (
        test_optimized_metrics[key] * 100
    )

    print(
        f"{label:<25}"
        f"{default_value:>17.2f}%"
        f"{optimized_value:>17.2f}%"
    )

print("-" * 61)

print(
    f"{'False Positive':<25}"
    f"{test_default_metrics['fp']:>18,}"
    f"{test_optimized_metrics['fp']:>18,}"
)

print(
    f"{'False Negative':<25}"
    f"{test_default_metrics['fn']:>18,}"
    f"{test_optimized_metrics['fn']:>18,}"
)

print(
    f"\nFalse positive reduction: "
    f"{fp_reduction * 100:.2f}%"
)


# ============================================================
# 9. THRESHOLD SENSITIVITY ANALYSIS
# ============================================================

print("\nGenerating threshold sensitivity analysis...")

thresholds = np.round(
    np.arange(
        SWEEP_MIN,
        SWEEP_MAX + SWEEP_STEP / 2,
        SWEEP_STEP
    ),
    3
)

sweep_results = []

for threshold in thresholds:

    result = calculate_metrics(
        y_test,
        p_test,
        float(threshold)
    )

    sweep_results.append(result)


# Save threshold sweep CSV.
csv_path = OUT_DIR / "threshold_sweep_test.csv"

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(sweep_results[0].keys())
    )

    writer.writeheader()
    writer.writerows(sweep_results)


# ============================================================
# 10. SAVE THRESHOLD SENSITIVITY GRAPH
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(
    thresholds,
    [m["fake_precision"] for m in sweep_results],
    label="Fake Precision",
    linewidth=2
)

ax.plot(
    thresholds,
    [m["fake_recall"] for m in sweep_results],
    label="Fake Recall",
    linewidth=2
)

ax.plot(
    thresholds,
    [m["fake_f1"] for m in sweep_results],
    label="Fake F1",
    linewidth=2
)

ax.axvline(
    DEFAULT_THRESHOLD,
    linestyle="--",
    linewidth=1.5,
    label="Default Threshold (0.50)"
)

ax.axvline(
    selected_threshold,
    linestyle=":",
    linewidth=2,
    label=f"Dev Selected ({selected_threshold:.4f})"
)

ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Metric Score")

ax.set_title(
    "Test-Set Operating-Point Sensitivity Analysis"
)

ax.set_ylim(0, 1)

ax.grid(True, alpha=0.3)

ax.legend()

fig.tight_layout()

sensitivity_path = (
    OUT_DIR / "threshold_sensitivity.png"
)

fig.savefig(
    sensitivity_path,
    dpi=220,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 11. CONFUSION MATRIX COMPARISON
# ============================================================

print("Generating confusion matrix comparison...")

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 4.5)
)

matrix_results = [
    (
        test_default_metrics,
        "Default Threshold: 0.50"
    ),
    (
        test_optimized_metrics,
        f"Optimized Threshold: {selected_threshold:.4f}"
    )
]

for ax, (result, title) in zip(
    axes,
    matrix_results
):

    matrix = np.array([
        [result["tn"], result["fp"]],
        [result["fn"], result["tp"]]
    ])

    ax.imshow(
        matrix,
        cmap="Blues"
    )

    ax.set_xticks(
        [0, 1],
        ["Predicted Real", "Predicted Fake"]
    )

    ax.set_yticks(
        [0, 1],
        ["Actual Real", "Actual Fake"]
    )

    ax.set_title(title)

    for (row, col), value in np.ndenumerate(matrix):

        ax.text(
            col,
            row,
            f"{value:,}",
            ha="center",
            va="center",
            fontsize=11
        )

fig.suptitle(
    "Confusion Matrix Comparison on the Same Test Set"
)

fig.tight_layout()

confusion_path = (
    OUT_DIR / "threshold_confusion_comparison.png"
)

fig.savefig(
    confusion_path,
    dpi=220,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 12. SAVE COMPLETE EXPERIMENT RESULTS
# ============================================================

summary = {

    "experiment": (
        "Development-Set-Constrained Operating "
        "Threshold Optimization for False-Positive Reduction"
    ),

    "selection_protocol": (
        "Maximize development fake precision subject to "
        "development fake recall >= 0.90"
    ),

    "dataset_information": {

        "development_samples": int(len(y_dev)),

        "development_chunks": int(dev_parts),

        "test_samples": int(len(y_test)),

        "test_chunks": int(test_parts),
    },

    "configuration": {

        "default_threshold": DEFAULT_THRESHOLD,

        "minimum_development_fake_recall": (
            MIN_DEV_FAKE_RECALL
        ),

        "selected_threshold": selected_threshold,
    },

    "development_metrics_at_selected_threshold": (
        best_dev_metrics
    ),

    "test_metrics_at_default_threshold": (
        test_default_metrics
    ),

    "test_metrics_at_optimized_threshold": (
        test_optimized_metrics
    ),

    "false_positive_reduction": {

        "default_false_positives": int(default_fp),

        "optimized_false_positives": int(optimized_fp),

        "reduction_count": int(default_fp - optimized_fp),

        "reduction_percentage": float(
            fp_reduction * 100
        ),
    },

    "notes": [

        "This experiment performs post-hoc decision threshold optimization.",

        "Model architecture and learned weights remain unchanged.",

        "Threshold selection uses development predictions only.",

        "The held-out test set is used only for final evaluation.",

        "Model scores are not calibrated probabilities.",

        "The optimized threshold introduces a precision-recall trade-off."
    ]
}

json_path = (
    OUT_DIR / "threshold_experiment_results.json"
)

with json_path.open(
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nSelected threshold:")
print(f"{selected_threshold:.8f}")

print("\nDefault threshold:", DEFAULT_THRESHOLD)

print("Optimized threshold:", selected_threshold)

print(
    "\nFalse positives reduced:",
    f"{default_fp:,} -> {optimized_fp:,}"
)

print(
    "False positive reduction:",
    f"{fp_reduction * 100:.2f}%"
)

print("\nGenerated files:")

print("1.", json_path)
print("2.", csv_path)
print("3.", sensitivity_path)
print("4.", confusion_path)

print("\nNo model retraining was performed.")

print("\n" + "=" * 65)
