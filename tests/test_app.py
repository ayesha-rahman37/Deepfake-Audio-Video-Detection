import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from app_components.validation import validate_uploaded_file
from app_components.pipeline import run_pipeline
from app_components.results import format_prediction


class FakeUpload:
    def __init__(self, name, content):
        self.name = name
        self.size = len(content)
        self.content = content


def test_empty_file():
    valid, message = validate_uploaded_file(
        FakeUpload("empty.mp4", b"")
    )
    assert not valid
    assert "empty" in message.lower()


def test_unsupported_format():
    valid, message = validate_uploaded_file(
        FakeUpload("document.pdf", b"sample")
    )
    assert not valid
    assert "unsupported" in message.lower()


def test_oversized_file():
    uploaded = FakeUpload(
        "large.mp4", b"x" * (200 * 1024 * 1024 + 1)
    )
    valid, message = validate_uploaded_file(uploaded)
    assert not valid
    assert "200 MB" in message


def test_valid_video_upload():
    valid, message = validate_uploaded_file(
        FakeUpload("sample.mp4", b"video-data")
    )
    assert valid
    assert message == "File is valid."


def test_processing_failure_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_pipeline(
            tmp_path / "missing.mp4",
            preprocess_fn=lambda path: path,
            inference_fn=lambda data: data,
        )


def test_processing_failure_from_preprocessing(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video-data")

    def broken_preprocessor(path):
        raise RuntimeError("Preprocessing failed")

    with pytest.raises(RuntimeError, match="Preprocessing failed"):
        run_pipeline(
            video,
            preprocess_fn=broken_preprocessor,
            inference_fn=lambda data: data,
        )


def test_format_prediction():
    result = format_prediction({
        "label": "Fake",
        "confidence": 0.95,
        "window_evidence": [{"window": 1, "score": 0.91}],
    })
    assert result["label"] == "Fake"
    assert result["confidence"] == 0.95
    assert len(result["window_evidence"]) == 1


def test_invalid_prediction_label():
    with pytest.raises(ValueError):
        format_prediction({
            "label": "Unknown",
            "confidence": 0.8,
            "window_evidence": [],
        })