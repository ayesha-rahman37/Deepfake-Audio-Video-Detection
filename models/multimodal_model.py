"""Integrated audio-video deepfake detector.

Inputs
------
video: [batch, frames, 3, 224, 224]
audio: [batch, 64, time_frames]

The ImageNet-pretrained ResNet50 remains frozen. The BiLSTM, audio CNN,
and reliability-aware fusion layers are trainable.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.visual_encoder import VisualEncoder
from models.temporal_model import TemporalModel
from models.audio_encoder import AudioEncoder
from models.fusion import ReliabilityAwareFusion


class MultimodalDeepfakeDetector(nn.Module):
    def __init__(
        self,
        pretrained_visual: bool = True,
        freeze_visual: bool = True,
        frame_batch_size: int = 4,
    ):
        super().__init__()
        if frame_batch_size < 1:
            raise ValueError("frame_batch_size must be at least 1")

        self.visual_encoder = VisualEncoder(
            pretrained=pretrained_visual,
            freeze_backbone=freeze_visual,
        )
        self.temporal_model = TemporalModel(input_dim=2048, hidden_dim=256)
        self.audio_encoder = AudioEncoder(embedding_dim=256)
        self.fusion = ReliabilityAwareFusion(
            video_dim=self.temporal_model.output_dim,
            audio_dim=256,
            fusion_dim=256,
        )
        self.freeze_visual = freeze_visual
        self.frame_batch_size = frame_batch_size

        if self.freeze_visual:
            self.visual_encoder.eval()

    def train(self, mode: bool = True):
        """Keep the frozen visual backbone in eval mode during training."""
        super().train(mode)
        if self.freeze_visual:
            self.visual_encoder.eval()
        return self

    def forward(self, video: torch.Tensor, audio: torch.Tensor):
        if video.ndim != 5:
            raise ValueError(
                "Expected video shape [batch, frames, 3, height, width], "
                f"got {tuple(video.shape)}"
            )
        if audio.ndim != 3:
            raise ValueError(
                "Expected audio shape [batch, 64, time_frames], "
                f"got {tuple(audio.shape)}"
            )
        if video.shape[0] != audio.shape[0]:
            raise ValueError("Audio and video batch sizes must match")

        batch_size, frame_count, channels, height, width = video.shape
        if channels != 3:
            raise ValueError(f"Expected 3 RGB channels, got {channels}")

        frames = video.reshape(batch_size * frame_count, channels, height, width)
        frame_features = []
        # Keep GPU memory low on consumer GPUs by encoding a few frames at a time.
        with torch.set_grad_enabled(not self.freeze_visual):
            for start in range(0, frames.shape[0], self.frame_batch_size):
                frame_chunk = frames[start : start + self.frame_batch_size]
                frame_features.append(self.visual_encoder(frame_chunk))
        frame_features = torch.cat(frame_features, dim=0)
        frame_features = frame_features.reshape(batch_size, frame_count, -1)
        video_embedding = self.temporal_model(frame_features)

        # Match the normalization used by the existing audio-only training script.
        audio = (audio + 40.0) / 40.0
        audio_embedding, audio_logit = self.audio_encoder(audio)

        fused_features, logits, reliability = self.fusion(
            video_embedding, audio_embedding
        )
        return {
            "logits": logits,
            "reliability": reliability,
            "video_embedding": video_embedding,
            "audio_embedding": audio_embedding,
            "fused_features": fused_features,
            "audio_logit": audio_logit,
        }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalDeepfakeDetector(pretrained_visual=False).to(device)
    model.eval()
    video = torch.randn(1, 16, 3, 224, 224, device=device)
    audio = torch.randn(1, 64, 313, device=device) * 10 - 40
    with torch.no_grad():
        result = model(video, audio)
    print("Integrated multimodal model test successful!")
    print("Device:", device)
    print("Video embedding:", tuple(result["video_embedding"].shape))
    print("Audio embedding:", tuple(result["audio_embedding"].shape))
    print("Logits:", tuple(result["logits"].shape))
    print("Reliability:", tuple(result["reliability"].shape))
