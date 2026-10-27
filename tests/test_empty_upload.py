
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_components.validation import validate_uploaded_file


class FakeUpload:
    def __init__(self, name, content):
        self.name = name
        self.size = len(content)


def test_empty_video_file_is_rejected():
    uploaded_file = FakeUpload("empty.mp4", b"")

    valid, message = validate_uploaded_file(uploaded_file)

    assert valid is False
    assert message == "The uploaded file is empty."


def test_missing_upload_is_rejected():
    valid, message = validate_uploaded_file(None)

    assert valid is False
    assert message == "Please upload a video file."
