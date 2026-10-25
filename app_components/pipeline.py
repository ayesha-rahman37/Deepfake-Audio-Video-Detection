
from pathlib import Path
from typing import Callable, Any


def run_pipeline(
    video_path: str | Path,
    preprocess_fn: Callable[[str], Any],
    inference_fn: Callable[[Any], Any],
) -> Any:
    """
    Run video preprocessing followed by model inference.

    The preprocessing and inference functions are supplied by
    the caller so this pipeline can connect to the project's
    existing preprocessing code and trained model.
    """
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    if not path.is_file():
        raise ValueError("The video path must point to a file.")

    if not callable(preprocess_fn):
        raise TypeError("preprocess_fn must be callable.")

    if not callable(inference_fn):
        raise TypeError("inference_fn must be callable.")

    preprocessed_data = preprocess_fn(str(path))
    prediction = inference_fn(preprocessed_data)

    return prediction
