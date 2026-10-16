# Deepfake Audio-Video Detection

A multimodal deep learning research project for detecting
real and deepfake videos using both visual and audio information.

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

## Dataset

Dataset files are not included in this repository.
Dataset access, licensing and preparation instructions
will be documented separately.

## Team Workflow

Development takes place in feature branches.
Changes are submitted through pull requests to `main`.

## Project Status

Environment setup and baseline implementation are in progress.