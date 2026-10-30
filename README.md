# Deepfake Audio-Video Detection

A multimodal deep learning research project for detecting real and deepfake videos using both visual and audio information.

## Proposed Methodology

- Full-frame video feature extraction
- CNN-based visual representation learning
- Temporal modeling using BiLSTM or GRU
- Audio processing using Log-Mel spectrograms and CNNs
- Baseline concatenation-based multimodal fusion
- Reliability-Aware Adaptive Multimodal Fusion
- Controlled robustness evaluation

## Project Structure

- `preprocessing/`: Video and audio preprocessing
- `datasets/`: Dataset loading and data management
- `models/`: Visual, audio and fusion models
- `training/`: Model training
- `evaluation/`: Metrics and experiment evaluation
- `configs/`: Experiment configuration
- `results/`: Final experiment results

## Requirements

- Python environment with the project's dependencies
- PyTorch and torchvision
- FFmpeg support through `imageio-ffmpeg`
- An NVIDIA GPU is recommended for model training; CPU execution may be slower.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ayesha-rahman37/Deepfake-Audio-Video-Detection.git
cd Deepfake-Audio-Video-Detection
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install common dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Install PyTorch and torchvision

Choose the installation command for your machine. Do not install both variants in the same environment.

**Windows or Linux with a compatible NVIDIA GPU**

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**CPU-only environment**

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Apple Silicon Mac**

```bash
python -m pip install torch torchvision
```

Check the official PyTorch installation selector if your operating system, Python version, or hardware requires a different build.

### 5. Configure the LAV-DF dataset path

The dataset is not included in this repository. Download it separately from its official source and follow its access and licensing terms.

By default, the loader expects the dataset at:

```text
D:\LAV-DF
```

To use a different location, set the `LAVDF_ROOT` environment variable before running the project.

**Windows PowerShell example**

```powershell
$env:LAVDF_ROOT = "E:\Datasets\LAV-DF"
```

**macOS / Linux example**

```bash
export LAVDF_ROOT="/path/to/LAV-DF"
```

The metadata CSV is expected at:

```text
data/metadata/lavdf_metadata.csv
```

Ensure the metadata paths match the downloaded dataset layout.

### 6. Test the dataset loader

```bash
python datasets/lavdf_dataset.py
```

The script loads one sample window and prints the video tensor shape, audio feature shape, label, and window duration.

## Dataset

Dataset files are not included in this repository. Do not upload raw dataset videos to GitHub. Dataset access, licensing and preparation instructions should follow the official dataset terms.

## Team Workflow

Development takes place in feature branches. Changes are submitted through pull requests to `main`.

## Project Status

Environment setup and baseline implementation are in progress. Model training and controlled evaluation are pending.

## Full-frame video preprocessing

`preprocessing.video_reader.VideoReader` decodes frames sequentially and yields
RGB images with the original stream index and timestamp provenance. OpenCV's
decoder-position timestamps are used when they advance reliably and are
normalized to the first frame; OpenCV does not guarantee these are exact source
presentation timestamps for every backend. Otherwise the reader explicitly
estimates timestamps from reported FPS. That fallback assumes a constant
frame rate and is not exact for variable-frame-rate video.
The reader's `metadata` is finalized after its iterator is exhausted and
contains the decoded duration estimate, reported metadata, and warnings.

`preprocessing.temporal_segmenter.segment_video(duration)` covers the video
using half-open `[start, end)` intervals, with five-second windows and no
overlap by default. `preprocessing.frame_sampler.sample_video(path)` makes a
metadata pass followed by a sequential sampling pass, choosing the nearest
decoded full frame to each of 16 evenly spaced bin-center timestamps per
window. It returns sampled RGB images plus timestamps, source frame indices,
segment boundaries, and warnings. If multiple targets resolve to one source
frame, that frame is returned once rather than duplicated.

The `VisualEncoder` accepts ImageNet-normalized RGB tensors with shape
`[batch, 3, height, width]` and returns `[batch, 2048]` ResNet50 features.
`preprocess_rgb` accepts unnormalized RGB tensors in `[0, 1]`, resizes the
whole frame to 224x224 by default (without cropping), and applies ImageNet
normalization. `TemporalModel` accepts ordered features shaped
`[batch, sequence, feature_dim]` and returns a pooled temporal representation;
optional sequence lengths exclude padded frames.