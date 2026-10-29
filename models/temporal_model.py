
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TemporalModel(nn.Module):
    """
    Learn temporal patterns from a sequence of visual features.

    Input:  [batch_size, frames, feature_dim]
    Output: [batch_size, hidden_dim * directions]
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, num_layers) < 1:
            raise ValueError("input_dim, hidden_dim, and num_layers must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1).")

        self.input_dim = input_dim
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(
        self,
        visual_features: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a masked mean of BiLSTM outputs as ``[batch, output_dim]``.

        ``lengths`` optionally gives each sequence's valid prefix length, which
        excludes padded frames from both recurrent processing and pooling.
        """
        if not isinstance(visual_features, torch.Tensor):
            raise TypeError("visual_features must be a torch.Tensor.")
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
        batch_size, sequence_length, _ = visual_features.shape
        if batch_size < 1 or sequence_length < 1:
            raise ValueError("Batch size and sequence length must be positive.")
        if not visual_features.is_floating_point():
            raise TypeError("visual_features must have a floating-point dtype.")
        if not torch.isfinite(visual_features).all():
            raise ValueError("visual_features must contain only finite values.")
        parameter = self.lstm.weight_ih_l0
        if visual_features.device != parameter.device:
            raise ValueError(
                f"visual_features are on {visual_features.device}, but model "
                f"parameters are on {parameter.device}."
            )
        if visual_features.dtype != parameter.dtype:
            raise TypeError(
                f"visual_features use {visual_features.dtype}, but model parameters "
                f"use {parameter.dtype}."
            )

        if lengths is None:
            sequence_output, _ = self.lstm(visual_features)
            valid_lengths = torch.full(
                (batch_size,),
                sequence_length,
                device=visual_features.device,
                dtype=torch.long,
            )
        else:
            if not isinstance(lengths, torch.Tensor) or lengths.ndim != 1:
                raise ValueError("lengths must be a 1D tensor with one value per batch.")
            if lengths.shape[0] != batch_size:
                raise ValueError("lengths must contain one value per batch item.")
            if lengths.is_floating_point() or lengths.dtype == torch.bool:
                raise TypeError("lengths must contain integer sequence lengths.")
            valid_lengths = lengths.to(device=visual_features.device, dtype=torch.long)
            if torch.any(valid_lengths < 1) or torch.any(valid_lengths > sequence_length):
                raise ValueError("Every sequence length must be within [1, sequence_length].")
            packed = pack_padded_sequence(
                visual_features,
                valid_lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = self.lstm(packed)
            sequence_output, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=sequence_length,
            )

        positions = torch.arange(sequence_length, device=visual_features.device)
        mask = positions.unsqueeze(0) < valid_lengths.unsqueeze(1)
        return (sequence_output * mask.unsqueeze(-1)).sum(dim=1) / valid_lengths.unsqueeze(1)


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
