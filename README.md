# Multimodal Deep Learning for Deepfake Audio-Video Detection

A research-oriented multimodal deep learning application that analyzes both visual and audio information to distinguish real videos from manipulated or deepfake videos.

The project combines spatial visual features, temporal modeling, audio representations, and reliability-aware multimodal fusion in a unified detection pipeline.

## Project Overview

Deepfake videos can manipulate facial appearance, expressions, lip movements, and audio. Analyzing only video frames or audio independently may miss important manipulation evidence.

This project uses both modalities:

- **Visual modality:** Extracts facial and frame-level visual representations.
- **Audio modality:** Extracts speech-related acoustic representations.
- **Temporal modeling:** Learns sequential patterns across video frames.
- **Multimodal fusion:** Combines visual and audio information using a reliability-aware fusion mechanism.

The application provides a Streamlit interface for uploading videos, running detection, and reviewing analysis results.

## Key Features

- Multimodal audio-video deepfake detection.
- ResNet50-based visual feature extraction.
- BiLSTM-based temporal feature modeling.
- CNN-based audio feature extraction.
- Log-Mel spectrogram representation.
- Reliability-aware multimodal fusion.
- Five-second video-window analysis.
- Sixteen sampled frames per video window.
- Real/fake prediction with model-generated scores.
- Per-window prediction evidence.
- Video upload validation.
- User registration and authentication.
- Detection history and saved results.
- Streamlit-based interactive dashboard.

## System Architecture

```text
                Input Video
                     |
          Video and Audio Extraction
                     |
          +----------+----------+
          |                     |
     Video Frames           Audio Signal
          |                     |
       ResNet50             Log-Mel
          |                 Spectrogram
       BiLSTM                   |
          |                 Audio CNN
          |                     |
          +----------+----------+
                     |
          Reliability-Aware Fusion
                     |
             Classification
                     |
              Real / Fake
                     |
          Prediction and Scores
                     |
             Streamlit UI
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Visual Encoder | ResNet50 |
| Temporal Model | BiLSTM |
| Audio Encoder | CNN |
| Audio Processing | Librosa |
| Video Processing | OpenCV |
| Audio Extraction | FFmpeg / imageio-ffmpeg |
| Multimodal Fusion | Reliability-Aware Fusion |
| Web Application | Streamlit |
| Database | SQLite |
| Evaluation | Scikit-learn |
| Testing | Pytest |

## Dataset

The project uses the **LAV-DF (Large-scale Audio-Visual Deepfake) dataset**.

Dataset splits:

- Training
- Development / Validation
- Testing

The dataset is not included in this repository. Users must obtain it separately and comply with its access and licensing conditions.

The application and training pipeline expect the dataset in the directory configured through the dataset loader.

Default location:

```text
D:\LAV-DF
```

For a custom location, configure the `LAVDF_ROOT` environment variable.

Windows PowerShell:

```powershell
$env:LAVDF_ROOT = "E:\Datasets\LAV-DF"
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ayesha-rahman37/Deepfake-Audio-Video-Detection.git

cd Deepfake-Audio-Video-Detection
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Install PyTorch and Torchvision

Install a compatible PyTorch build for your hardware.

For NVIDIA GPU systems, select the appropriate CUDA installation command from the official PyTorch installation selector.

For CPU-only systems:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 5. Configure the dataset

Ensure that the LAV-DF dataset and metadata are correctly configured before training or evaluation.

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application provides user authentication, video upload, analysis, prediction results, and detection history.

## Model Training

The multimodal model can be trained using:

```bash
python training/train_multimodal.py
```

A small pilot experiment can be run with:

```bash
python training/train_multimodal.py --max-train-samples 512 --max-val-samples 128 --epochs 3 --batch-size 1
```

The pilot configuration is intended for experimentation and debugging. It must not be presented as final research performance.

For reproducibility, training uses configurable random seeds, dataset splits, and checkpoint metadata.

## Model Evaluation

Run the evaluation script:

```bash
python evaluation/evaluate_multimodal.py
```

The evaluation pipeline supports resumable chunk-based processing of the held-out test split.

Evaluation metrics include:

- Accuracy
- Balanced Accuracy
- Fake Precision
- Fake Recall
- Fake F1-score
- Macro F1-score
- ROC-AUC
- Confusion Matrix

## Reported Evaluation Results

The previously completed full-test evaluation reported the following results:

| Metric | Result |
|---|---:|
| Accuracy | 68.24% |
| Balanced Accuracy | 72.80% |
| Fake Precision | 55.84% |
| Fake Recall | 94.72% |
| Fake F1-score | 70.26% |
| Macro F1-score | 68.09% |
| ROC-AUC | 79.14% |

Confusion Matrix:

```text
                 Predicted
                 Real    Fake

Actual Real     17353   16754
Actual Fake      1181   21184
```

These results belong to the previously evaluated checkpoint and experiment. They are not the results of the later three-epoch pilot run.

**Important:** The reported scores are experimental results, not proof that every prediction is correct. The relatively low fake precision indicates that some real videos may be incorrectly classified as fake.

## Testing

Run the automated test suite:

```bash
python -m pytest -q
```

Run Python compilation checks:

```bash
python -m compileall -q app.py app_components inference models training evaluation
```

The latest local test run completed with:

```text
80 passed
```

The compilation check completed without reported errors.

## Project Structure

```text
Deepfake-Audio-Video-Detection/
│
├── app.py
│
├── app_components/
│   ├── auth.py
│   ├── theme.py
│   └── validation.py
│
├── configs/
│   └── config.yaml
│
├── datasets/
│   └── lavdf_dataset.py
│
├── models/
│   ├── audio_encoder.py
│   ├── fusion.py
│   ├── multimodal_model.py
│   ├── temporal_model.py
│   └── visual_encoder.py
│
├── preprocessing/
│
├── inference/
│   └── predict_video.py
│
├── training/
│   └── train_multimodal.py
│
├── evaluation/
│   ├── evaluate_multimodal.py
│   ├── inspect_checkpoint.py
│   └── map_error_samples.py
│
├── checkpoints/
├── results/
├── tests/
├── data/
├── docs/
├── requirements.txt
└── README.md
```

## Limitations

- Model predictions are not guaranteed to be correct.
- Performance depends on dataset quality and distribution.
- The current experimental model can produce false-positive predictions.
- Scores should not be interpreted as calibrated probabilities.
- Generalization to unseen manipulation techniques requires further evaluation.
- Full training and evaluation may require substantial GPU memory and processing time.

## Future Improvements

- Increase training data and training duration.
- Improve visual encoder fine-tuning.
- Reduce false-positive predictions.
- Improve cross-dataset generalization.
- Conduct controlled comparisons between fusion strategies.
- Evaluate robustness against different manipulation techniques.
- Improve score calibration and explainability.

## Research Disclaimer

This project is developed for academic and research purposes. It is not intended to serve as a standalone forensic verification system or as the sole basis for high-stakes decisions.

## Contributors

Developed as an academic machine learning research project.

---

**Repository:** [Deepfake Audio-Video Detection](https://github.com/ayesha-rahman37/Deepfake-Audio-Video-Detection)
