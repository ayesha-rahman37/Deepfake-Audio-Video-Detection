
from app_components.validation import validate_uploaded_file


class FakeUpload:
    def __init__(self, name, size):
        self.name = name
        self.size = size


def test_file_over_200_mb_is_rejected():
    uploaded_file = FakeUpload(
        "large.mp4",
        200 * 1024 * 1024 + 1
    )

    valid, message = validate_uploaded_file(uploaded_file)

    assert valid is False
    assert message == "File size must be 200 MB or less."


def test_file_at_200_mb_is_accepted():
    uploaded_file = FakeUpload(
        "maximum.mp4",
        200 * 1024 * 1024
    )

    valid, message = validate_uploaded_file(uploaded_file)

    assert valid is True
    assert message == "File is valid."
