
import pytest

from app_components.pipeline import run_pipeline


def test_pipeline_runs_preprocessing_and_inference(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"sample video")

    def preprocess(path):
        assert path == str(video)
        return {"frames": [1, 2, 3]}

    def inference(data):
        assert data == {"frames": [1, 2, 3]}
        return {"label": "FAKE", "confidence": 0.95}

    result = run_pipeline(video, preprocess, inference)

    assert result == {"label": "FAKE", "confidence": 0.95}


def test_pipeline_missing_file():
    def preprocess(path):
        return path

    def inference(data):
        return data

    with pytest.raises(FileNotFoundError):
        run_pipeline(
            "missing_video.mp4",
            preprocess,
            inference,
        )


def test_pipeline_rejects_directory(tmp_path):
    def preprocess(path):
        return path

    def inference(data):
        return data

    with pytest.raises(ValueError):
        run_pipeline(tmp_path, preprocess, inference)


def test_pipeline_rejects_invalid_preprocessor(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"sample video")

    def inference(data):
        return data

    with pytest.raises(TypeError):
        run_pipeline(video, None, inference)


def test_pipeline_rejects_invalid_inference_function(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"sample video")

    def preprocess(path):
        return path

    with pytest.raises(TypeError):
        run_pipeline(video, preprocess, None)


def test_pipeline_propagates_processing_error(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"sample video")

    def preprocess(path):
        raise RuntimeError("Preprocessing failed")

    def inference(data):
        return data

    with pytest.raises(RuntimeError, match="Preprocessing failed"):
        run_pipeline(video, preprocess, inference)
