"""Integrated full-frame audio-video deepfake detector.

Input video: [B, frames, 3, H, W] (ImageNet-normalized RGB frames)
Input audio: [B, 64, time_frames] (log-Mel values in dB)

Use fusion_mode='concat' for the basic baseline and 'reliability' for the
proposed quality-aware adaptive fusion model.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.visual_encoder import VisualEncoder
from models.temporal_model import TemporalModel
from models.audio_encoder import AudioEncoder
from models.fusion import ConcatenationFusion, ReliabilityAwareFusion


class MultimodalDeepfakeDetector(nn.Module):
    def __init__(self, pretrained_visual: bool = True,
                 freeze_visual: bool = True, frame_batch_size: int = 4,
                 fusion_mode: str = "reliability"):
        super().__init__()
        if frame_batch_size < 1:
            raise ValueError("frame_batch_size must be at least 1")
        if fusion_mode not in {"concat", "reliability"}:
            raise ValueError("fusion_mode must be 'concat' or 'reliability'")

        self.visual_encoder = VisualEncoder(
            pretrained=pretrained_visual, freeze_backbone=freeze_visual
        )
        self.temporal_model = TemporalModel(input_dim=2048, hidden_dim=256)
        self.audio_encoder = AudioEncoder(embedding_dim=256)
        if fusion_mode == "concat":
            self.fusion = ConcatenationFusion(video_dim=512, audio_dim=256)
        else:
            self.fusion = ReliabilityAwareFusion(
                video_dim=512, audio_dim=256, fusion_dim=256, quality_dim=4
            )

        self.fusion_mode = fusion_mode
        self.freeze_visual = freeze_visual
        self.frame_batch_size = frame_batch_size
        if self.freeze_visual:
            self.visual_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_visual:
            self.visual_encoder.eval()
        return self

    @staticmethod
    def extract_quality_indicators(video: torch.Tensor,
                                   audio_log_mel: torch.Tensor) -> torch.Tensor:
        """Return four deterministic, imperfect input-quality proxies.

        Order: visual edge/sharpness proxy, visual mean luminance, fraction of
        log-Mel cells below -60 dB, normalized log-Mel spread. These indicators
        are not ground-truth quality labels or proof of manipulation.
        """
        if video.ndim != 5 or audio_log_mel.ndim != 3:
            raise ValueError("Expected video [B,T,3,H,W] and audio [B,64,F]")
        if video.shape[0] != audio_log_mel.shape[0]:
            raise ValueError("Audio and video batch sizes must match")

        # Restore approximate RGB values from the ImageNet normalization.
        mean = video.new_tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        std = video.new_tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        rgb = (video * std + mean).clamp(0.0, 1.0)
        gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1]
                + 0.114 * rgb[:, :, 2])
        horizontal = (gray[..., :, 1:] - gray[..., :, :-1]).abs().mean(dim=(1, 2, 3))
        vertical = (gray[..., 1:, :] - gray[..., :-1, :]).abs().mean(dim=(1, 2, 3))
        sharpness_proxy = ((horizontal + vertical) * 0.5)
        sharpness_proxy = sharpness_proxy / (sharpness_proxy + 0.1)
        luminance_mean = gray.mean(dim=(1, 2, 3))

        low_energy_fraction = (audio_log_mel < -60.0).float().mean(dim=(1, 2))
        log_mel_spread = (audio_log_mel.std(dim=(1, 2), unbiased=False) / 40.0).clamp(0, 1)
        quality = torch.stack(
            (sharpness_proxy, luminance_mean, low_energy_fraction, log_mel_spread),
            dim=1,
        )
        return quality.detach()

    def forward(self, video: torch.Tensor, audio: torch.Tensor):
        if video.ndim != 5:
            raise ValueError(
                "Expected video [B, frames, 3, H, W], "
                f"got {tuple(video.shape)}"
            )
        if audio.ndim != 3:
            raise ValueError(
                "Expected audio [B, 64, time_frames], "
                f"got {tuple(audio.shape)}"
            )
        if video.shape[0] != audio.shape[0]:
            raise ValueError("Audio and video batch sizes must match")
        if video.shape[2] != 3:
            raise ValueError(f"Expected RGB channel dimension 3, got {video.shape[2]}")
        if audio.shape[1] != 64:
            raise ValueError(f"Expected 64 Mel bands, got {audio.shape[1]}")

        quality = self.extract_quality_indicators(video, audio)
        batch_size, frame_count, channels, height, width = video.shape
        frames = video.reshape(batch_size * frame_count, channels, height, width)

        frame_features = []
        with torch.set_grad_enabled(not self.freeze_visual):
            for start in range(0, frames.shape[0], self.frame_batch_size):
                frame_features.append(
                    self.visual_encoder(frames[start:start + self.frame_batch_size])
                )
        frame_features = torch.cat(frame_features, dim=0)
        frame_features = frame_features.reshape(batch_size, frame_count, -1)
        video_embedding = self.temporal_model(frame_features)

        # Maintain the normalization used by the existing audio training script.
        normalized_audio = (audio + 40.0) / 40.0
        audio_embedding, audio_logit = self.audio_encoder(normalized_audio)
        fused, logits, reliability = self.fusion(
            video_embedding, audio_embedding, quality
        )

        return {
            "logits": logits,
            "reliability": reliability,
            "quality_features": quality,
            "video_embedding": video_embedding,
            "audio_embedding": audio_embedding,
            "fused_features": fused,
            "audio_logit": audio_logit,
        }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video = torch.randn(1, 16, 3, 224, 224, device=device)
    audio = torch.randn(1, 64, 313, device=device) * 10.0 - 40.0
    for mode in ("concat", "reliability"):
        model = MultimodalDeepfakeDetector(
            pretrained_visual=False, fusion_mode=mode
        ).to(device).eval()
        with torch.no_grad():
            output = model(video, audio)
        assert output["logits"].shape == (1,)
        assert output["reliability"].shape == (1, 2)
        assert torch.isfinite(output["logits"]).all()
        print(
            f"{mode} model PASS | device={device} "
            f"video={tuple(output['video_embedding'].shape)} "
            f"audio={tuple(output['audio_embedding'].shape)} "
            f"fused={tuple(output['fused_features'].shape)} "
            f"logits={tuple(output['logits'].shape)} "
            f"quality={tuple(output['quality_features'].shape)}"
        )
