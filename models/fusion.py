
import torch
import torch.nn as nn


class ReliabilityAwareFusion(nn.Module):
    """
    Reliability-aware fusion of audio and video representations.

    Video input: [batch_size, 512]
    Audio input: [batch_size, 256]

    Outputs:
        fused_features: [batch_size, 256]
        logits: [batch_size]
        reliability: [batch_size, 2]
    """

    def __init__(self, video_dim=512, audio_dim=256, fusion_dim=256,
                 dropout=0.3):
        super().__init__()

        self.video_projection = nn.Sequential(
            nn.Linear(video_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.audio_projection = nn.Sequential(
            nn.Linear(audio_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Estimate the relative reliability of each modality.
        self.reliability_estimator = nn.Sequential(
            nn.Linear(fusion_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, video_features, audio_features):
        if video_features.ndim != 2:
            raise ValueError("Video features must have shape [batch, 512].")

        if audio_features.ndim != 2:
            raise ValueError("Audio features must have shape [batch, 256].")

        if video_features.shape[0] != audio_features.shape[0]:
            raise ValueError("Audio and video batch sizes must match.")

        video = self.video_projection(video_features)
        audio = self.audio_projection(audio_features)

        combined = torch.cat([video, audio], dim=1)

        # Sum of the two reliability weights is 1.
        reliability = torch.softmax(
            self.reliability_estimator(combined), dim=1
        )

        video_weight = reliability[:, 0:1]
        audio_weight = reliability[:, 1:2]

        fused_features = (
            video_weight * video + audio_weight * audio
        )

        logits = self.classifier(fused_features).squeeze(1)

        return fused_features, logits, reliability


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ReliabilityAwareFusion().to(device)
    model.eval()

    video_features = torch.randn(4, 512, device=device)
    audio_features = torch.randn(4, 256, device=device)

    with torch.no_grad():
        fused, logits, reliability = model(
            video_features, audio_features
        )

    print("Fusion model test successful!")
    print(f"Device: {device}")
    print(f"Video input: {tuple(video_features.shape)}")
    print(f"Audio input: {tuple(audio_features.shape)}")
    print(f"Fused features: {tuple(fused.shape)}")
    print(f"Prediction logits: {tuple(logits.shape)}")
    print(f"Reliability weights: {tuple(reliability.shape)}")
    print("Reliability weight sums:", reliability.sum(dim=1).tolist())
