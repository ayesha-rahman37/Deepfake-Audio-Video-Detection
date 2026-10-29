
"""Video preprocessing and inference pipeline."""

from pathlib import Path

from app_components.inference import analyze_video


def run_pipeline(
    video_path,
    preprocess_fn=None,
    inference_fn=None,
    checkpoint_path=None,
):
    """Validate a video, preprocess it, and run inference.

    Optional preprocessing and inference functions allow the pipeline
    to be tested without loading a trained model.
    """

    path = Path(video_path)

    # Check that the video path exists.
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    # Check that the path points to a file.
    if not path.is_file():
        raise ValueError(f"Video path is not a file: {path}")

    # Reject empty video files.
    if path.stat().st_size == 0:
        raise ValueError("The video file is empty.")

    # Test mode: use the supplied preprocessing and inference functions.
    if preprocess_fn is not None or inference_fn is not None:
        if preprocess_fn is None:
            raise TypeError(
                "preprocess_fn must be a callable function."
            )

        if inference_fn is None:
            raise TypeError(
                "inference_fn must be a callable function."
            )

        if not callable(preprocess_fn):
            raise TypeError(
                "preprocess_fn must be a callable function."
            )

        if not callable(inference_fn):
            raise TypeError(
                "inference_fn must be a callable function."
            )

        processed_data = preprocess_fn(str(path))
        result = inference_fn(processed_data)

        return result

    # Production mode: run the actual trained-model inference pipeline.
    if checkpoint_path is None:
        return analyze_video(str(path))

    return analyze_video(
        str(path),
        checkpoint_path=checkpoint_path,
    )


def run_video_pipeline(video_path, checkpoint_path=None):
    """Backward-compatible entry point for the Streamlit app."""

    return run_pipeline(
        video_path,
        checkpoint_path=checkpoint_path,
    )
