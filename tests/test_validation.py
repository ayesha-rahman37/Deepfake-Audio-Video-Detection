
import pytest

from app_components.validation import validate_uploaded_file


class MockUploadedFile:
    def __init__(self, name, size):
        self.name = name
        self.size = size


def test_empty_upload():
    valid, message = validate_uploaded_file(None)

    assert valid is False
    assert message == "Please upload a video file."


def test_unsupported_format():
    file = MockUploadedFile("document.pdf", 1000)

    valid, message = validate_uploaded_file(file)

    assert valid is False
    assert message == "Unsupported video format."


def test_file_too_large():
    file = MockUploadedFile("video.mp4", 201 * 1024 * 1024)

    valid, message = validate_uploaded_file(file)

    assert valid is False
    assert message == "File size must be 200 MB or less."


def test_zero_byte_file():
    file = MockUploadedFile("video.mp4", 0)

    valid, message = validate_uploaded_file(file)

    assert valid is False
    assert message == "The uploaded file is empty."


def test_valid_video():
    file = MockUploadedFile("video.mp4", 1024 * 1024)

    valid, message = validate_uploaded_file(file)

    assert valid is True
    assert message == "File is valid."


def test_uppercase_extension():
    file = MockUploadedFile("video.MP4", 1024)

    valid, message = validate_uploaded_file(file)

    assert valid is True
    assert message == "File is valid."
