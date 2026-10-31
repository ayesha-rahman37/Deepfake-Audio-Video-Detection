"""Audio-video fusion modules for the deepfake detection project.

Fusion modes
------------
- ConcatenationFusion: reproducible baseline [video; audio] -> MLP.
- ReliabilityAwareFusion: quality-aware modality gating followed by weighted
  feature concatenation, as specified by the project blueprint.

The reliability weights are learned gates, not calibrated probabilities.
"""

import torch
import torch.nn as nn


class ConcatenationFusion(nn.Module):
    """Basic audio-video feature-concatenation baseline."""

    def __init__(self, video_dim: int = 512, audio_dim: int = 256,
                 hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.output_dim = video_dim + audio_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, video_features: torch.Tensor,
                audio_features: torch.Tensor,
                quality_features: torch.Tensor | None = None):
        _validate_features(video_features, audio_features)
        fused = torch.cat((video_features, audio_features), dim=1)
        logits = self.classifier(fused).squeeze(1)
        # A non-adaptive baseline has no learned reliability gate. Use neutral
        # values only to keep the integrated model's output schema consistent.
        reliability = fused.new_full((fused.shape[0], 2), 0.5)
        return fused, logits, reliability


class ReliabilityAwareFusion(nn.Module):
    """Quality-aware adaptive fusion using weighted feature concatenation.

    Inputs:
        video_features: [batch, video_dim]
        audio_features: [batch, audio_dim]
        quality_features: [batch, quality_dim], e.g. video sharpness,
            mean brightness, log-Mel floor ratio, and log-Mel spread proxy.

    Output:
        fused_features: [batch, 2 * fusion_dim]
        logits: [batch]
        reliability: [batch, 2] (video/audio weights, sum to one)
    """

    def __init__(self, video_dim: int = 512, audio_dim: int = 256,
                 fusion_dim: int = 256, quality_dim: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        if min(video_dim, audio_dim, fusion_dim, quality_dim) < 1:
            raise ValueError("All feature dimensions must be positive")

        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.fusion_dim = fusion_dim
        self.quality_dim = quality_dim
        self.output_dim = fusion_dim * 2

        self.video_projection = nn.Sequential(
            nn.Linear(video_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.reliability_estimator = nn.Sequential(
            nn.Linear(fusion_dim * 2 + quality_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, video_features: torch.Tensor,
                audio_features: torch.Tensor,
                quality_features: torch.Tensor | None = None):
        _validate_features(video_features, audio_features)
        batch_size = video_features.shape[0]

        if quality_features is None:
            quality_features = video_features.new_zeros(
                (batch_size, self.quality_dim)
            )
        if quality_features.ndim != 2 or quality_features.shape != (
            batch_size, self.quality_dim
        ):
            raise ValueError(
                f"Expected quality_features shape {(batch_size, self.quality_dim)}, "
                f"got {tuple(quality_features.shape)}"
            )
        quality_features = quality_features.to(
            device=video_features.device, dtype=video_features.dtype
        )

        video = self.video_projection(video_features)
        audio = self.audio_projection(audio_features)
        gate_input = torch.cat((video, audio, quality_features), dim=1)
        reliability = torch.softmax(self.reliability_estimator(gate_input), dim=1)

        video_weight = reliability[:, 0:1]
        audio_weight = reliability[:, 1:2]
        # Blueprint requires gated features to be concatenated, not averaged.
        fused = torch.cat((video_weight * video, audio_weight * audio), dim=1)
        logits = self.classifier(fused).squeeze(1)
        return fused, logits, reliability


def _validate_features(video_features: torch.Tensor,
                       audio_features: torch.Tensor) -> None:
    if video_features.ndim != 2:
        raise ValueError(
            "Video features must have shape [batch, video_dim], "
            f"got {tuple(video_features.shape)}"
        )
    if audio_features.ndim != 2:
        raise ValueError(
            "Audio features must have shape [batch, audio_dim], "
            f"got {tuple(audio_features.shape)}"
        )
    if video_features.shape[0] != audio_features.shape[0]:
        raise ValueError("Audio and video batch sizes must match")
    if video_features.device != audio_features.device:
        raise ValueError("Audio and video features must be on the same device")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video = torch.randn(4, 512, device=device)
    audio = torch.randn(4, 256, device=device)
    quality = torch.rand(4, 4, device=device)

    for name, module in (
        ("concatenation", ConcatenationFusion()),
        ("reliability-aware", ReliabilityAwareFusion()),
    ):
        module = module.to(device).eval()
        with torch.no_grad():
            fused, logits, reliability = module(video, audio, quality)
        assert fused.shape[0] == 4 and logits.shape == (4,)
        assert torch.allclose(
            reliability.sum(dim=1), torch.ones(4, device=device), atol=1e-6
        )
        print(
            f"{name} fusion PASS | fused={tuple(fused.shape)} "
            f"logits={tuple(logits.shape)} reliability={tuple(reliability.shape)}"
        )
