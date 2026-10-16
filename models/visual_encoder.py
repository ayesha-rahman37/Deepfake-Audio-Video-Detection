
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class VisualEncoder(nn.Module):
    """Extract visual features from full RGB video frames."""

    def __init__(self, pretrained=True, freeze_backbone=True):
        super().__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)

        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, frames):
        """
        Args:
            frames: Tensor [batch, 3, height, width]

        Returns:
            Features: Tensor [batch, 2048]
        """
        return self.backbone(frames)


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = VisualEncoder(
        pretrained=True,
        freeze_backbone=True,
    ).to(device)
    model.eval()

    # Small test batch to fit the laptop GPU memory.
    dummy_frames = torch.randn(2, 3, 224, 224, device=device)

    with torch.no_grad():
        features = model(dummy_frames)

    print("Visual encoder test successful!")
    print(f"Device: {device}")
    print(f"Input shape: {tuple(dummy_frames.shape)}")
    print(f"Feature shape: {tuple(features.shape)}")
    print(f"Feature dimension: {model.feature_dim}")
