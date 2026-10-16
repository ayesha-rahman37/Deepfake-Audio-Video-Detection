
import torch
import torch.nn as nn


class TemporalModel(nn.Module):
    """
    Learn temporal patterns from a sequence of visual features.

    Input:  [batch_size, frames, feature_dim]
    Output: [batch_size, hidden_dim * 2]
    """

    def __init__(
        self,
        input_dim=2048,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
    ):
        super().__init__()

        self.input_dim = input_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_dim = hidden_dim * 2

    def forward(self, visual_features):
        if visual_features.ndim != 3:
            raise ValueError(
                "Expected [batch, frames, features], "
                f"got {tuple(visual_features.shape)}"
            )

        if visual_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected feature dimension {self.input_dim}, "
                f"got {visual_features.shape[-1]}"
            )

        sequence_output, _ = self.lstm(visual_features)

        # Average the temporal representations of all frames.
        video_representation = sequence_output.mean(dim=1)

        return video_representation


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = TemporalModel().to(device)
    model.eval()

    # Simulate 16 frames, each represented by 2048 features.
    dummy_features = torch.randn(2, 16, 2048, device=device)

    with torch.no_grad():
        output = model(dummy_features)

    print("Temporal model test successful!")
    print(f"Device: {device}")
    print(f"Input shape: {tuple(dummy_features.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output dimension: {model.output_dim}")
