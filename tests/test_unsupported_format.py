
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


def test_pdf_file_is_rejected():
    uploaded_file = FakeUpload("document.pdf", b"sample content")

    valid, message = validate_uploaded_file(uploaded_file)

    assert valid is False
    assert message == "Unsupported video format."


def test_txt_file_is_rejected():
    uploaded_file = FakeUpload("document.txt", b"sample content")

    valid, message = validate_uploaded_file(uploaded_file)

    assert valid is False
    assert message == "Unsupported video format."


def test_uppercase_unsupported_extension_is_rejected():
    uploaded_file = FakeUpload("document.PDF", b"sample content")

    valid, message = validate_uploaded_file(uploaded_file)

    assert valid is False
    assert message == "Unsupported video format."
