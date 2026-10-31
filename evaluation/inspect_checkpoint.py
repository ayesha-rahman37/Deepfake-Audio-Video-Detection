
from pathlib import Path
import torch


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "checkpoints" / "multimodal_reliability_best.pt"


def load_checkpoint():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
        )

    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint format.")

    return checkpoint


def inspect_checkpoint():
    print("\n" + "=" * 65)
    print("MULTIMODAL DEEPFAKE MODEL - CHECKPOINT INSPECTION")
    print("=" * 65)

    checkpoint = load_checkpoint()

    print("\n[1] CHECKPOINT METADATA")
    print("-" * 45)

    for key, value in checkpoint.items():
        if key == "model_state_dict":
            continue
        print(f"{key}: {value}")

    if "model_state_dict" not in checkpoint:
        raise KeyError("model_state_dict is missing!")

    state_dict = checkpoint["model_state_dict"]

    print("\n[2] MODEL WEIGHT SUMMARY")
    print("-" * 45)

    print("Total parameter tensors:", len(state_dict))

    total_parameters = 0
    visual_parameters = 0
    audio_parameters = 0
    temporal_parameters = 0
    fusion_parameters = 0

    visual_keys = []
    non_finite_tensors = []

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        total_parameters += tensor.numel()
        lower_name = name.lower()

        if any(word in lower_name for word in [
            "visual", "backbone", "resnet"
        ]):
            visual_parameters += tensor.numel()
            visual_keys.append(name)

        if "audio" in lower_name:
            audio_parameters += tensor.numel()

        if any(word in lower_name for word in [
            "temporal", "lstm"
        ]):
            temporal_parameters += tensor.numel()

        if any(word in lower_name for word in [
            "fusion", "gate", "quality"
        ]):
            fusion_parameters += tensor.numel()

        if tensor.is_floating_point():
            if not torch.isfinite(tensor).all():
                non_finite_tensors.append(name)

    print(f"Total parameters: {total_parameters:,}")
    print(f"Visual-related parameters: {visual_parameters:,}")
    print(f"Audio-related parameters: {audio_parameters:,}")
    print(f"Temporal-related parameters: {temporal_parameters:,}")
    print(f"Fusion-related parameters: {fusion_parameters:,}")

    print("\n[3] VISUAL ENCODER WEIGHT CHECK")
    print("-" * 45)

    print("Visual-related tensors found:", len(visual_keys))

    for name in visual_keys[:20]:
        tensor = state_dict[name]
        print(f"{name}: {tuple(tensor.shape)}")

    if len(visual_keys) > 20:
        print(f"... and {len(visual_keys) - 20} more")

    if not visual_keys:
        print("WARNING: No visual-related keys matched the search.")
        print("Inspect the complete model key list below.")

    print("\n[4] IMPORTANT MODEL KEYS")
    print("-" * 45)

    for name, tensor in list(state_dict.items())[:35]:
        if isinstance(tensor, torch.Tensor):
            print(f"{name}: {tuple(tensor.shape)}")
        else:
            print(f"{name}: {type(tensor).__name__}")

    print("\n[5] NON-FINITE WEIGHT CHECK")
    print("-" * 45)

    if non_finite_tensors:
        print("WARNING: Non-finite values found:")
        for name in non_finite_tensors[:20]:
            print(name)
    else:
        print("PASS: No NaN or Inf values found in floating weights.")

    print("\n[6] FUSION MODE CHECK")
    print("-" * 45)

    fusion_mode = checkpoint.get("fusion_mode")
    print("Saved fusion mode:", fusion_mode)

    if fusion_mode == "reliability":
        print("PASS: Reliability fusion checkpoint.")
    elif fusion_mode is None:
        print("WARNING: Fusion mode metadata is missing.")
    else:
        print("WARNING: Unexpected fusion mode.")

    print("\n" + "=" * 65)
    print("INSPECTION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    inspect_checkpoint()
