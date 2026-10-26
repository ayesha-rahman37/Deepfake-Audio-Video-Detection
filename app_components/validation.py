
from pathlib import Path

MAX_FILE_SIZE_MB = 200

SUPPORTED_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
}


def validate_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return False, "Please upload a video file."

    extension = Path(uploaded_file.name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        return False, "Unsupported video format."

    file_size_mb = uploaded_file.size / (1024 * 1024)

    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, "File size must be 200 MB or less."

    if uploaded_file.size == 0:
        return False, "The uploaded file is empty."

    return True, "File is valid."
