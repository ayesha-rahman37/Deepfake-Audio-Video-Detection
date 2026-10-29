import pytest
import torch

from models.temporal_model import TemporalModel
from models.visual_encoder import VisualEncoder


def test_visual_encoder_returns_per_frame_resnet_features():
    model = VisualEncoder(pretrained=False, freeze_backbone=True).eval()
    frames = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        features = model(frames)
    assert features.shape == (2, 2048)
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())


def test_visual_encoder_preprocesses_full_rgb_images():
    model = VisualEncoder(pretrained=False, image_size=64).eval()
    images = torch.rand(2, 3, 32, 48)
    processed = model.preprocess_rgb(images)
    assert processed.shape == (2, 3, 64, 64)
    assert torch.isfinite(processed).all()


def test_visual_encoder_unfrozen_backbone_has_gradients():
    model = VisualEncoder(pretrained=False, freeze_backbone=False, image_size=64).train()
    frames = torch.randn(2, 3, 64, 64)
    model(frames).sum().backward()
    assert model.backbone.conv1.weight.grad is not None


@pytest.mark.parametrize(
    "invalid",
    [torch.randn(3, 224, 224), torch.randn(2, 1, 224, 224)],
)
def test_visual_encoder_rejects_invalid_frame_dimensions(invalid):
    model = VisualEncoder(pretrained=False)
    with pytest.raises(ValueError):
        model(invalid)


def test_temporal_model_batch_one_output_and_gradients():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalModel(input_dim=12, hidden_dim=5, num_layers=1).to(device)
    features = torch.randn(1, 4, 12, device=device, requires_grad=True)
    representation = model(features)
    assert representation.shape == (1, 10)
    representation.sum().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_temporal_model_masks_padding_for_variable_lengths():
    model = TemporalModel(input_dim=8, hidden_dim=4).eval()
    features = torch.randn(2, 6, 8)
    lengths = torch.tensor([6, 3])
    representation = model(features, lengths=lengths)
    changed_padding = features.clone()
    changed_padding[1, 3:] = 10000
    changed_representation = model(changed_padding, lengths=lengths)
    assert representation.shape == (2, 8)
    assert torch.allclose(representation[1], changed_representation[1], atol=1e-6)


def test_temporal_model_uses_sequence_order():
    model = TemporalModel(input_dim=8, hidden_dim=4).eval()
    features = torch.randn(1, 5, 8)
    forward_representation = model(features)
    reverse_representation = model(features.flip(dims=(1,)))
    assert not torch.allclose(forward_representation, reverse_representation)


def test_temporal_model_rejects_invalid_inputs():
    model = TemporalModel(input_dim=8, hidden_dim=4)
    with pytest.raises(ValueError):
        model(torch.randn(2, 8))
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 7))
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 8), lengths=torch.tensor([4, 2]))
