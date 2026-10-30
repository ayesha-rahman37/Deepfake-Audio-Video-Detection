
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class VisualEncoder(nn.Module):
    """Extract 2048-dimensional per-frame ResNet50 features.

    Forward inputs follow the existing detector contract: float RGB tensors
    ``[batch, 3, height, width]`` already normalized with ImageNet statistics.
    ``preprocess_rgb`` is provided for unnormalized RGB tensors in ``[0, 1]``.
    Resizing covers the full frame (no center crop or face-specific transform).
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 1:
            raise ValueError("image_size must be a positive integer.")

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)

        self.feature_dim = backbone.fc.in_features
        self.image_size = image_size
        self.input_mean = (0.485, 0.456, 0.406)
        self.input_std = (0.229, 0.224, 0.225)
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def preprocess_rgb(self, frames: torch.Tensor) -> torch.Tensor:
        """Resize full RGB frames in ``[0, 1]`` and apply ImageNet normalization."""
        self._validate_input(frames)
        if not frames.is_floating_point():
            raise TypeError("RGB frames must have a floating-point dtype.")
        if not torch.isfinite(frames).all():
            raise ValueError("RGB frames must contain only finite values.")
        if frames.min() < 0 or frames.max() > 1:
            raise ValueError("preprocess_rgb expects RGB values in [0, 1].")
        resized = F.interpolate(
            frames,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = resized.new_tensor(self.input_mean).view(1, 3, 1, 1)
        std = resized.new_tensor(self.input_std).view(1, 3, 1, 1)
        return (resized - mean) / std

    @staticmethod
    def _validate_input(frames: torch.Tensor) -> None:
        if not isinstance(frames, torch.Tensor):
            raise TypeError("frames must be a torch.Tensor.")
        if frames.ndim != 4:
            raise ValueError(
                "Expected frames [batch, 3, height, width], "
                f"got {tuple(frames.shape)}"
            )
        if frames.shape[0] < 1 or frames.shape[1] != 3:
            raise ValueError("frames must contain a non-empty batch of RGB images.")
        if frames.shape[2] < 1 or frames.shape[3] < 1:
            raise ValueError("Frame height and width must be positive.")

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: ImageNet-normalized float tensor [batch, 3, height, width]

        Returns:
            Features: Tensor [batch, 2048]
        """
        self._validate_input(frames)
        if not frames.is_floating_point():
            raise TypeError("frames must have a floating-point dtype.")
        if not torch.isfinite(frames).all():
            raise ValueError("frames must contain only finite values.")
        parameter = next(self.backbone.parameters())
        if frames.device != parameter.device:
            raise ValueError(
                f"frames are on {frames.device}, but model parameters are on "
                f"{parameter.device}."
            )
        if frames.dtype != parameter.dtype:
            raise TypeError(
                f"frames use {frames.dtype}, but model parameters use {parameter.dtype}."
            )
        if frames.shape[-2:] != (self.image_size, self.image_size):
            frames = F.interpolate(
                frames,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
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
