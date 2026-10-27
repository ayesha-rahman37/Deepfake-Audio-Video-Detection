
from pathlib import Path

import streamlit as st

from app_components.validation import validate_uploaded_file

st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🎥",
    layout="wide",
)

st.title("🎥 Deepfake Audio-Video Detection")
st.write(
    "Upload a video to validate the file and check "
    "whether a trained prediction model is available."
)

checkpoint_dir = Path("checkpoints")
checkpoint_files = [
    path
    for path in checkpoint_dir.rglob("*")
    if path.is_file()
    and path.suffix.lower() in {".pt", ".pth", ".ckpt", ".onnx"}
] if checkpoint_dir.exists() else []

uploaded_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "mov", "avi", "mkv", "webm"],
)

if uploaded_file is not None:
    valid, message = validate_uploaded_file(uploaded_file)

    if not valid:
        st.error(message)
    else:
        st.success(message)
        st.video(uploaded_file)

        if not checkpoint_files:
            st.warning(
                "No trained model checkpoint was found. "
                "A reliable Real/Fake prediction is not available yet."
            )

        if st.button("Process Video", type="primary"):
            if not checkpoint_files:
                st.error(
                    "Processing stopped: provide a compatible trained "
                    "model checkpoint before running inference."
                )
            else:
                st.warning(
                    "A checkpoint exists, but model loading and inference "
                    "are not connected yet."
                )
