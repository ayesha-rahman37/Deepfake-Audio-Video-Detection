```python
from pathlib import Path

import streamlit as st

from app_components.validation import validate_uploaded_file

st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🎥",
    layout="wide",
)

# Styling
st.markdown(
    """
    <style>
    .stApp {
        background: #f4f7fb;
        color: #182230;
    }
    .block-container {
        max-width: 1120px;
        padding-top: 1.5rem;
    }
    .hero {
        background: linear-gradient(120deg, #ffffff, #eaf1ff);
        border: 1px solid #d8e4ff;
        border-radius: 18px;
        padding: 25px;
        margin-bottom: 22px;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: #182230;
    }
    .hero-copy {
        color: #475467;
        line-height: 1.6;
    }
    .stButton > button {
        background: #2457d6;
        color: white;
        border-radius: 10px;
        min-height: 44px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Find model checkpoint files
checkpoint_dir = Path("checkpoints")
checkpoint_files = (
    [
        path
        for path in checkpoint_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".pt", ".pth", ".ckpt", ".onnx"}
    ]
    if checkpoint_dir.exists()
    else []
)

# Sidebar
with st.sidebar:
    st.markdown("## 🎬 AI Detection Lab")
    st.caption("Multimodal research prototype")
    st.divider()
    st.markdown("### Workspace")
    st.write("📁 Video analysis")
    st.write("🖼️ Visual analysis")
    st.write("🔊 Audio analysis")
    st.write("🧠 Multimodal prediction")
    st.divider()
    st.markdown("### Model status")

    if checkpoint_files:
        st.success(f"{len(checkpoint_files)} checkpoint file(s) found")
    else:
        st.warning("No checkpoint file found")

    st.caption(
        "A checkpoint file alone does not confirm that inference is ready."
    )

# Header
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🎥 Deepfake Audio-Video Detection</div>
        <p class="hero-copy">
            Upload and preview a video. Model-based detection requires
            a connected preprocessing and inference pipeline.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Overview
col1, col2, col3 = st.columns(3)
col1.metric("Input", "Video upload")
col2.metric("Maximum size", "200 MB")
col3.metric("Model assets", "Found" if checkpoint_files else "Not found")

# Upload section
st.subheader("01 · Upload video")
st.caption("Supported formats: MP4, MOV, AVI, MKV and WEBM.")

uploaded_file = st.file_uploader(
    "Choose a video",
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
            with st.status(
                "Starting video processing...",
                expanded=True,
            ) as status:
                try:
                    st.write("Step 1: Upload validation completed.")

                    if not checkpoint_files:
                        raise FileNotFoundError(
                            "No compatible model checkpoint was found "
                            "in the checkpoints folder."
                        )

                    st.write("Step 2: Checking model availability.")

                    raise RuntimeError(
                        "Model loading and inference are not connected "
                        "to the application yet."
                    )

                except FileNotFoundError as error:
                    status.update(
                        label="Processing failed",
                        state="error",
                    )
                    st.error(str(error))

                except Exception as error:
                    status.update(
                        label="Processing failed",
                        state="error",
                    )
                    st.error(
                        f"Video processing could not be completed: {error}"
                    )

st.divider()
st.caption(
    "AI Detection Lab · Research Prototype · Predictions require "
    "a connected trained inference pipeline."
)