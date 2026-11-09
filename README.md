# Multimodal Deep Learning for Deepfake Audio-Video Detection

A research prototype for detecting manipulated videos by analyzing
aligned visual and audio information. The system processes videos in
**5-second windows**, samples **16 full-frame images per window**,
extracts Log-Mel audio features, and combines temporal visual and
acoustic representations using multimodal fusion.

> **Research disclaimer:** Model scores are not calibrated probabilities
> and are not forensic proof of authenticity. Predictions should not be
> used as the sole basis for high-stakes decisions.

## 1. Project overview

The project investigates whether combining audio and video evidence
improves deepfake detection compared with single-modality and
traditional machine-learning baselines.

### Model components

-   **Visual branch:** ResNet50 frame encoder followed by a
    bidirectional LSTM temporal model.
-   **Audio branch:** Log-Mel spectrogram representation followed by a
    CNN audio encoder.
-   **Fusion baselines:** Feature concatenation and reliability-aware
    fusion.
-   **Reliability-aware fusion:** Learns modality weights using modality
    representations and quality-related features. These weights are
    model gates, not calibrated confidence values.
-   **Traditional ML:** SVM (RBF) and Random Forest classifiers trained
    on frozen multimodal embeddings and quality features.
-   **Inference:** Processes an input video in windows and saves
    video-level and window-level predictions as JSON.

### High-level workflow

``` text
Input video
    |
    +-----------------------------+
    |                             |
Full-frame video              Audio stream
    |                             |
16 frames / 5-second window   Timestamp-aligned segment
    |                             |
ResNet50                       Log-Mel spectrogram
    |                             |
BiLSTM                         Audio CNN
    |                             |
    +-------------+---------------+
                  |
       Concatenation / Reliability fusion
                  |
           Real / Fake score
                  |
       Window-level and video-level output
```

## 2. Dataset and preprocessing

The experiments use the **LAV-DF (Large-scale Audio-Visual Deepfake)
dataset**.

Dataset splits: - Train - Development (dev) - Test

The dataset is not included in this repository. Obtain it separately and
comply with its access and licensing requirements.

The dataset loader reads the metadata CSV and expects the video files
under the configured LAV-DF root. The default path in the current loader
is `D:\LAV-DF`. A custom path can be provided through `LAVDF_ROOT`.

Windows PowerShell example:

``` powershell
$env:LAVDF_ROOT = "E:\Datasets\LAV-DF"
```

The current dataset indexing audit reported:

  Split     Indexed windows
  ------- -----------------
  Train             170,460
  Dev                70,220
  Test               56,472

The recorded leakage audit found zero original-video overlap between
train, dev, and test. The loader reports missing files and invalid
metadata rows rather than silently hiding them. Missing or undecodable
audio is not silently replaced with synthetic silence.

## 3. Architecture and methods

  Component                 Method
  ------------------------- ----------------------------------
  Video frame encoder       ResNet50
  Temporal video model      2-layer bidirectional LSTM
  Audio representation      64-bin Log-Mel spectrogram
  Audio encoder             CNN
  Basic fusion              Concatenation
  Proposed fusion           Reliability-aware learned gating
  Traditional ML            RBF-SVM and Random Forest
  Deep-learning framework   PyTorch

The dataset labels are window-level labels derived from overlap with
annotated fake intervals. The video-only label additionally requires
video manipulation to be present in the source record.

## 4. Experimental results

Results below come from different experiment sizes and must **not** be
compared as if they used one identical test set. Subset results are
explicitly identified.

### 4.1 Reliability-aware multimodal model --- full test split

  Metric                Result
  ------------------- --------
  Accuracy              68.24%
  Balanced accuracy     72.80%
  Fake precision        55.84%
  Fake recall           94.72%
  Fake F1-score         70.26%
  Macro F1-score        68.09%
  ROC-AUC               79.14%

Confusion matrix:

``` text
                 Predicted
                 Real    Fake
Actual Real     17353   16754
Actual Fake      1181   21184
```

The high fake recall is accompanied by a substantial number of false
positives. This limitation must be considered when interpreting
predictions.

### 4.2 Audio-only candidate --- 2,048 test samples

  Metric             Result
  ---------------- --------
  Accuracy           85.40%
  Fake precision     72.42%
  Fake recall        77.82%
  Fake F1-score      75.02%
  ROC-AUC            91.14%

This is a subset evaluation of the audio candidate checkpoint, not a
full-test result.

### 4.3 Traditional ML on frozen multimodal embeddings

Feature vector: 512-dimensional video embedding + 256-dimensional audio
embedding + 4 quality features. Training used 2,048 windows and testing
used 512 windows.

  Classifier        Accuracy   Macro F1   ROC-AUC
  --------------- ---------- ---------- ---------
  RBF-SVM             79.69%     79.59%    88.80%
  Random Forest       79.69%     79.63%    87.80%

These are subset-based embedding-classifier results. They are not
raw-pixel classical baselines or full-test results.

### 4.4 Multimodal comparison and ablation

A 256-window balanced test subset was used for the recorded comparison:

  Model / setting                       Accuracy   Macro F1   ROC-AUC
  ----------------------------------- ---------- ---------- ---------
  Concatenation, full                     66.41%     65.07%    69.82%
  Reliability fusion, full                72.66%     71.38%    80.84%
  Reliability fusion, video ablated       71.88%     70.35%    80.79%
  Reliability fusion, audio ablated       50.00%     33.33%    35.05%

Ablation was performed by removing a modality embedding at the fusion
stage. It should not be described as a separately trained video-only or
audio-only model. The small change after video ablation and large
degradation after audio ablation suggest that this checkpoint relies
more heavily on audio evidence.

### 4.5 Video-only baseline

The evaluated video-only checkpoint predicted every sample as real on a
balanced 256-window test subset:

-   Accuracy: 50.00%
-   Fake precision, recall, and F1: 0
-   ROC-AUC: 48.11%

This is a weak baseline and must be reported transparently; it is not
evidence of successful video-only detection.

### 4.6 Robustness experiments

A preliminary 256-window benchmark was run with low-resolution, blur,
frame-drop, audio-noise, audio-quantization, audio/video shift, and
combined perturbations.

The recorded low-resolution, blur, and frame-drop metrics were identical
to baseline. Although the script contains tensor transformations for
these conditions, identical metrics mean their effective impact was not
demonstrated in this run. Treat these conditions as **inconclusive**,
not as proof of robustness.

Audio noise, audio/video shift, and combined perturbations caused
substantial performance degradation in the recorded run. These are
preliminary tensor-level perturbations, not a complete re-encoding study
using real-world codecs.

## 5. Inference

The tested inference command is:

``` powershell
python inference/predict_video.py --video "path\to\video.mp4" --output results\prediction.json
```

Example:

``` powershell
python inference/predict_video.py --video "C:\Videos\sample.mp4" --output results\prediction.json
```

The tested run processed a 179.1-second video as 36 windows and saved
`results/prediction.json`.

The JSON output contains: - Video-level prediction and fake/real
scores - Duration and processing device - Per-window start and end
timestamps - Per-window prediction scores - Learned video/audio
reliability gates

A successful inference run confirms that the pipeline executes; it does
not independently verify whether the prediction is correct. Validate
with labeled data when measuring accuracy.

## 6. Installation and environment

Create and activate a virtual environment:

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

``` powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install **PyTorch and Torchvision separately** using the official
PyTorch installation selector. Choose a build compatible with your
operating system, Python version, and hardware. For an NVIDIA GPU,
select the appropriate CUDA build. Do not assume that installing
`requirements.txt` has installed PyTorch or Torchvision.

Before running inference, obtain the trained checkpoint separately and
place it at:

``` text
checkpoints/multimodal_reliability_best.pt
```

See Section 19 for checkpoint details. To run dataset-dependent training
or evaluation scripts, configure `LAVDF_ROOT` as described in Section 2.

After installing the dependencies and placing the checkpoint, launch the
Streamlit application from the project root:

``` powershell
python -m streamlit run app.py
```
Or, streamlit run app.py

## 7. Evaluation scripts and outputs

Important scripts:

  ------------------------------------------------------------------------------------
  Script                                           Purpose
  ------------------------------------------------ -----------------------------------
  `evaluation/evaluate_multimodal.py`              Multimodal split evaluation

  `evaluation/evaluate_video_only_checkpoint.py`   Video-only checkpoint evaluation

  `evaluation/model_comparison_ablation.py`        Fusion comparison and modality
                                                   ablation

  `evaluation/robustness_benchmark.py`             Tensor-level robustness benchmark

  `evaluation/traditional_ml_baseline.py`          SVM and Random Forest embedding
                                                   baseline

  `inference/predict_video.py`                     Video inference and JSON output
  ------------------------------------------------------------------------------------

Key result files are stored under `results/`, including: -
`multimodal_reliability_test_metrics_full.json` -
`multimodal_reliability_dev_metrics_full.json` -
`model_comparison_ablation_test_256.json` -
`robustness_benchmark_reliability_test.json` -
`traditional_ml_baseline.json` - `audio_candidate_test_metrics.json` -
`video_only_v2_test_256_metrics.json` - `prediction.json`

Use the exact split, sample count, seed, checkpoint, and configuration
when reproducing or comparing experiments. Do not compare subset metrics
directly with full-test metrics.

## 8. Limitations

1.  The video-only checkpoint evaluated here is weak and predicted all
    samples as real in the tested subset.
2.  The full-test reliability model has a high false-positive count.
3.  The traditional ML and several modality comparison results use
    subsets.
4.  Robustness results for low-resolution, blur, and frame-drop are
    inconclusive because their recorded metrics matched baseline.
5.  Cross-dataset generalization has not been established.
6.  The model scores are not calibrated probabilities.
7.  Window-level classification does not provide exact manipulation
    timestamps unless ground-truth temporal intervals are used for a
    dedicated localization evaluation.
8.  A model prediction is not forensic proof and should not be used as
    the sole basis for consequential decisions.

## 9. Optional future work

-   Improve video-only representation learning and evaluate it on the
    same fixed test indices.
-   Validate perturbation strength by measuring input changes and
    testing actual transcoded videos.
-   Add calibration analysis (ECE/Brier score) if calibrated confidence
    is required.
-   Evaluate cross-dataset generalization.
-   Investigate transformer-based temporal models and temporal
    localization.

## 10. Research disclaimer

This is an academic research prototype. Predictions can be wrong, and
performance may change on videos from different sources, codecs,
manipulation methods, or recording conditions. Always interpret results
with the reported evaluation protocol and limitations.

## 11. Streamlit application and decision policy

The repository includes a Streamlit interface (`app.py`) with account
registration/login, video upload, multimodal analysis, visual/audio
evidence views, detection history, and saved detection reports. The
application calls `inference/predict_video.py` and stores detection
records in a local SQLite database. The database is local application
data and should not be committed or shared.

### Decision policies

The application provides two decision policies.

**Conservative Policy** - Fake score \<= 0.40: REAL - Fake score \> 0.40
and \< 0.60: INCONCLUSIVE - Fake score \>= 0.60: FAKE

This policy preserves an uncertainty region around the decision
boundary. An INCONCLUSIVE result means the score is near the decision
boundary and should not be forced into a binary conclusion.

**Optimized Threshold Policy** - Threshold: 0.51775074 - Fake score \<
0.51775074: REAL - Fake score \>= 0.51775074: FAKE

The optimized threshold was selected using the development split to
maximize fake precision while maintaining fake recall of at least 90%.

These are post-hoc decision rules. They do not retrain the model, modify
model weights, or calibrate its scores. Detailed test-set comparisons
are provided in Section 17.

Run the application from the project root:

``` powershell
python -m streamlit run app.py
```

## 12. Additional audit and reproducibility notes

The dataset manifest audit is saved as
`results/dataset_manifest_audit.json` when generated. Keep the audit
alongside the experiment outputs so the indexed split sizes, skipped
metadata/files, duration corrections, and sample-ID overlap checks can
be reviewed. The audit's sample-ID overlap check is distinct from the
original-video leakage audit described in Section 2.

A repaired/re-encoded sample video may be used for local pipeline
testing, but do not commit user-provided videos or other private media
to the repository. Store local SQLite databases, temporary media,
datasets, checkpoints, virtual environments, and machine-specific files
outside version control.

## 13. Temporal-Order Ablation

A preliminary temporal-order ablation was run on a balanced 256-window
test subset by comparing normal and reversed frame sequences.

-   Normal-order and reversed-order metrics were identical.
-   Mean absolute fake-score change: approximately 8.3e-08.
-   Prediction flips: 0.

This indicates that the evaluated checkpoint was effectively insensitive
to frame-order reversal under this test setup. It does not demonstrate
successful temporal reasoning.

## 14. Error Analysis

The full-test error-only export was reviewed in `ERROR_ANALYSIS.md` and
summarized in `results/error_analysis_summary.json`.

-   False positives: 16,754 windows.
-   False negatives: 1,181 windows.
-   Total misclassified windows: 17,935.

False positives substantially outnumber false negatives. The error-only
export cannot independently reproduce the full evaluation metrics.

## 15. Remaining Research Limitations

1.  Investigate and improve temporal-order sensitivity, since the
    current checkpoint showed negligible prediction changes under
    frame-order reversal; also improve video-only performance.
2.  Calibrate model scores using a dedicated calibration set.
3.  Evaluate cross-dataset generalization.
4.  Validate robustness using actual re-encoded videos.
5.  Reduce the substantial false-positive burden.
6.  Further investigate forensic reliability.

## 16. Application Upload Configuration

The Streamlit application supports video uploads up to 500 MB.

The upload size is configured in `.streamlit/config.toml`:

``` toml
[server]
maxUploadSize = 500
```

The application validates uploaded file extensions, empty files, and
maximum file size before processing.

## 17. Decision Operating Points

The application provides two decision policies.

### Conservative Policy

-   Fake score \<= 0.40: REAL
-   Fake score \>= 0.60: FAKE
-   Between 0.40 and 0.60: INCONCLUSIVE

This policy preserves an uncertainty region around the decision
boundary.

### Optimized Threshold Policy

-   Threshold: 0.51775074
-   Fake score below threshold: REAL
-   Fake score at or above threshold: FAKE

The threshold was selected using the development split to maximize fake
precision while maintaining fake recall of at least 90%.

This is post-hoc operating-point optimization. It does not retrain the
model, modify model weights, or calibrate its scores.

### Test-set comparison

  Metric                Default 0.50   Optimized 0.51775
  ------------------- -------------- -------------------
  Accuracy                    68.24%              71.54%
  Balanced Accuracy           72.80%              74.64%
  Fake Precision              55.84%              59.32%
  Fake Recall                 94.72%              89.55%
  Fake F1                     70.26%              71.37%
  ROC-AUC                     79.14%              79.14%

The optimized operating point reduced false positives by 3,019 (18.02%),
while fake recall decreased by 5.17 percentage points.

These results demonstrate a precision-recall trade-off rather than an
improvement in the underlying learned model.

## 18. Application Architecture

The user-facing application is implemented using Streamlit.

It includes: - User registration and login - Video upload and
validation - Multimodal video analysis - Conservative and optimized
decision policies - Window-level evidence - Detection history - Saved
detection reports - Local SQLite persistence

The application uses `inference/predict_video.py` for video analysis.

## 19. Model Checkpoint Setup

Model checkpoint files are excluded from Git version control.

Place the trained reliability-aware checkpoint at the path expected by
the inference implementation:

`checkpoints/multimodal_reliability_best.pt`

The checkpoint must be obtained separately before running inference on a
fresh installation.

## 20. Current Project Status and Future Work

The project includes an implemented Streamlit application and a
multimodal deepfake detection research pipeline. The application
provides an interface for running the model, but this does not remove
the research limitations described in this README.

Remaining research directions include: - Improving video-only
representation learning. - Investigating and improving temporal-order
sensitivity, since the current checkpoint showed negligible prediction
changes under frame-order reversal. - Calibrating model scores using a
dedicated calibration set. - Evaluating cross-dataset generalization. -
Testing robustness on genuinely re-encoded videos. - Reducing false
positives and improving forensic reliability.

The application is an academic research prototype. Its predictions are
not forensic proof and should not be used as the sole basis for
high-stakes decisions.
