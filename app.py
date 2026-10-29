from pathlib import Path
import tempfile

import streamlit as st

from app_components.validation import validate_uploaded_file
from app_components.inference import analyze_video
from app_components.results import format_prediction


# -------------------- PAGE SETTINGS --------------------

st.set_page_config(
    page_title="Deepfake Detection | AI Research Lab",
    page_icon="🎬",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

checkpoint_files = [
    file
    for file in CHECKPOINT_DIR.rglob("*")
    if file.is_file()
    and file.suffix.lower() in {".pt", ".pth", ".ckpt"}
    and file.stat().st_size > 0
] if CHECKPOINT_DIR.exists() else []


# -------------------- CUSTOM DESIGN --------------------

st.markdown(
    """
    <style>
    .stApp {
        background: #f4f7fb;
        color: #182230;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #d9e1ed;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 1.5rem;
    }

    .hero {
        background: linear-gradient(120deg, #ffffff, #eaf1ff);
        border: 1px solid #d8e4ff;
        border-radius: 18px;
        padding: 28px;
        margin-bottom: 22px;
    }

    .eyebrow {
        color: #2457d6;
        font-size: 12px;
        font-weight: bold;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .hero-title {
        color: #182230;
        font-size: 32px;
        font-weight: 800;
        margin-top: 8px;
    }

    .hero-copy {
        color: #475467;
        font-size: 16px;
        line-height: 1.6;
    }

    .section-title {
        font-size: 21px;
        font-weight: 750;
        color: #182230;
        margin-top: 20px;
    }

    .panel {
        background: #ffffff;
        border: 1px solid #dce3ee;
        border-radius: 15px;
        padding: 18px;
        margin-bottom: 16px;
    }

    .stButton > button {
        min-height: 44px;
        border-radius: 10px;
        font-weight: 700;
        background: #2457d6;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown("## 🎬 AI Detection Lab")
    st.caption("Multimodal research prototype")
    st.divider()

    st.markdown("### Workspace")
    st.markdown("📹 Video upload and preview")
    st.markdown("🖼️ Video frame processing")
    st.markdown("🔊 Audio feature extraction")
    st.markdown("🧠 PyTorch model inference")

    st.divider()
    st.markdown("### Model status")

    if checkpoint_files:
        st.success("Trained checkpoint file found")
    else:
        st.warning("No trained checkpoint found")

    st.caption(
        "A compatible trained checkpoint is required "
        "to generate a prediction."
    )


# -------------------- PAGE HEADER --------------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI-powered media research</div>
        <div class="hero-title">
            Deepfake Audio-Video Detection
        </div>
        <p class="hero-copy">
            Upload a video, review the preview and file information,
            and run the connected PyTorch inference pipeline.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------- OVERVIEW --------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Input type", "Video")

with col2:
    st.metric("Maximum file size", "200 MB")

with col3:
    st.metric(
        "Model checkpoint",
        "Available" if checkpoint_files else "Missing",
    )


# -------------------- VIDEO UPLOAD --------------------

st.markdown(
    '<div class="section-title">01 · Upload video</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Supported formats: MP4, MOV, AVI, MKV and WEBM."
)

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "mov", "avi", "mkv", "webm"],
    help="Maximum allowed file size is 200 MB.",
)


# -------------------- VALIDATION AND PREVIEW --------------------

if uploaded_file is None:
    st.info("Upload a video to view its details and preview.")

else:
    is_valid, message = validate_uploaded_file(uploaded_file)

    if not is_valid:
        st.error(message)

    else:
        st.success("Video uploaded and validated successfully.")

        file_size_mb = uploaded_file.size / (1024 * 1024)
        file_format = Path(uploaded_file.name).suffix.upper().lstrip(".")

        st.markdown(
            '<div class="section-title">02 · Video preview</div>',
            unsafe_allow_html=True,
        )

        st.video(uploaded_file)

        st.markdown("#### File information")

        info1, info2, info3 = st.columns(3)

        with info1:
            st.metric("File size", f"{file_size_mb:.2f} MB")

        with info2:
            st.metric("Format", file_format)

        with info3:
            st.metric("Validation", "Passed")


        # -------------------- MODEL STATUS --------------------

        st.markdown(
            '<div class="section-title">03 · Detection</div>',
            unsafe_allow_html=True,
        )

        if not checkpoint_files:
            st.warning(
                "No trained checkpoint was found. Upload and preview "
                "are available, but a Real/Fake prediction cannot be "
                "generated until a compatible trained checkpoint is added."
            )

        else:
            st.info(
                "A checkpoint was found. Its compatibility will be "
                "checked during inference."
            )


        # -------------------- VIDEO ANALYSIS --------------------

        if st.button(
            "Run video analysis",
            type="primary",
            use_container_width=True,
        ):

            status_box = st.status(
                "Starting video analysis...",
                expanded=True,
            )

            temp_video_path = None

            try:
                status_box.write("Preparing uploaded video...")

                extension = Path(uploaded_file.name).suffix.lower()

                with tempfile.NamedTemporaryFile(
                    suffix=extension,
                    delete=False,
                ) as temp_video:

                    temp_video.write(uploaded_file.getvalue())
                    temp_video_path = Path(temp_video.name)

                status_box.update(
                    label="Processing video and audio...",
                    state="running",
                )

                status_box.write(
                    "Extracting video frames and audio features, "
                    "then running PyTorch inference."
                )

                raw_prediction = analyze_video(temp_video_path)

                prediction = format_prediction(raw_prediction)

                status_box.update(
                    label="Analysis completed successfully.",
                    state="complete",
                )


                # -------------------- PREDICTION RESULT --------------------

                st.markdown("#### Prediction result")

                result_col, confidence_col = st.columns(2)

                with result_col:
                    if prediction["label"] == "Fake":
                        st.error("Prediction: Fake")
                    else:
                        st.success("Prediction: Real")

                with confidence_col:
                    st.metric(
                        "Confidence",
                        f'{prediction["confidence"] * 100:.2f}%',
                    )


                # -------------------- WINDOW-WISE EVIDENCE --------------------

                evidence = prediction["window_evidence"]

                if evidence:
                    st.markdown("#### Window-wise evidence")

                    st.dataframe(
                        evidence,
                        use_container_width=True,
                        hide_index=True,
                    )

                else:
                    st.info(
                        "The model did not return window-wise evidence."
                    )

                st.caption(
                    "Inference device: "
                    + str(raw_prediction.get("device", "Unknown"))
                )


            # -------------------- ERROR HANDLING --------------------

            except Exception as exc:
                status_box.update(
                    label="Video analysis failed.",
                    state="error",
                )

                st.error(f"Analysis failed: {exc}")

                st.info(
                    "No prediction was generated. Check the trained "
                    "checkpoint, preprocessing steps and model inference."
                )


            # -------------------- TEMPORARY FILE CLEANUP --------------------

            finally:
                if temp_video_path is not None:
                    try:
                        temp_video_path.unlink(missing_ok=True)
                    except OSError:
                        pass


# -------------------- FOOTER --------------------

st.divider()

st.caption(
    "AI Detection Lab · Research Prototype · "
    "Predictions require a compatible trained checkpoint."
)