
from pathlib import Path
import os
import tempfile

import streamlit as st
from app_components.validation import validate_uploaded_file
from inference.predict_video import predict_video


# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Deepfake Detection | AI Research Lab",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

checkpoint_files = (
    [
        p for p in CHECKPOINT_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".pt", ".pth", ".ckpt", ".onnx"}
    ]
    if CHECKPOINT_DIR.exists()
    else []
)


# --------------------------------------------------
# LIGHT THEME UI
# --------------------------------------------------

st.markdown(
    """
    <style>
    :root {
        --ink: #182230;
        --muted: #475467;
        --line: #d9e1ed;
        --blue: #2457d6;
        --page: #f4f7fb;
    }

    .stApp {
        background: var(--page);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(244,247,251,.96);
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] * {
        color: #263449;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 1.25rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4 {
        color: #182230 !important;
        letter-spacing: -0.025em;
    }

    p, li, label {
        color: #344054;
    }

    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        color: #475467 !important;
    }

    .hero {
        background: linear-gradient(120deg, #ffffff, #eaf1ff);
        border: 1px solid #d8e4ff;
        border-radius: 20px;
        padding: 25px 28px;
        margin-bottom: 20px;
    }

    .eyebrow {
        color: #2457d6;
        font-size: .75rem;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .hero-title {
        color: #182230;
        font-size: clamp(1.7rem, 3vw, 2.35rem);
        font-weight: 800;
        line-height: 1.18;
        margin-bottom: 9px;
    }

    .hero-copy {
        color: #475467;
        font-size: .98rem;
        line-height: 1.6;
        margin: 0;
        max-width: 720px;
    }

    .section-title {
        color: #182230;
        font-size: 1.15rem;
        font-weight: 800;
        margin: 18px 0 4px;
    }

    .section-copy {
        color: #475467;
        font-size: .9rem;
        margin-bottom: 12px;
    }

    .status-card {
        background: #ffffff;
        border: 1px solid #dce3ee;
        border-radius: 14px;
        padding: 15px 17px;
        min-height: 88px;
        margin-bottom: 14px;
        box-shadow: 0 2px 7px rgba(16,24,40,.035);
    }

    .status-label {
        color: #475467;
        font-size: .76rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: .055em;
        margin-bottom: 7px;
    }

    .status-value {
        color: #182230;
        font-size: 1.02rem;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    .panel {
        background: #ffffff;
        border: 1px solid #dce3ee;
        border-radius: 16px;
        padding: 18px;
        margin: 10px 0 16px;
        box-shadow: 0 3px 10px rgba(16,24,40,.035);
    }

    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #a9bad8;
        border-radius: 14px;
        padding: 10px 12px;
    }

    [data-testid="stFileUploader"] * {
        color: #263449;
    }

    [data-testid="stVideo"] {
        width: 100%;
        max-width: 640px;
        margin: 0 auto 8px;
        border-radius: 13px;
        overflow: hidden;
        background: #111827;
    }

    [data-testid="stVideo"] video {
        display: block;
        width: 100%;
        max-height: 280px;
        object-fit: contain;
        background: #111827;
        border-radius: 13px;
    }

    .stButton > button,
    .stButton > button[kind="primary"] {
        min-height: 45px;
        border-radius: 11px;
        font-weight: 750;
        background: #2457d6 !important;
        border: 1px solid #2457d6 !important;
        color: #ffffff !important;
    }

    .stButton > button p,
    .stButton > button[kind="primary"] p {
        color: #ffffff !important;
        font-weight: 750 !important;
    }

    .stButton > button:hover {
        background: #1745b5 !important;
        border-color: #1745b5 !important;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dce3ee;
        border-radius: 13px;
        padding: 12px 15px;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        color: #475467 !important;
        font-size: .82rem !important;
    }

    [data-testid="stMetricValue"] {
        color: #182230 !important;
        font-size: 1.35rem !important;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    [data-testid="stAlert"] p {
        color: #263449 !important;
    }

    .footer {
        border-top: 1px solid #dce3ee;
        padding-top: 16px;
        margin-top: 28px;
        color: #596579;
        text-align: center;
        font-size: .82rem;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem .85rem 1.5rem;
        }

        .hero {
            padding: 20px 17px;
            border-radius: 15px;
        }

        [data-testid="stVideo"] {
            max-width: 100%;
        }

        [data-testid="stVideo"] video {
            max-height: 220px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🎬 AI Detection Lab")
    st.caption("Multimodal research prototype")
    st.divider()

    st.markdown("### Workspace")
    st.markdown("📁 **Video analysis**")
    st.caption("Upload and preview a video.")
    st.markdown("🖼️ **Visual analysis**")
    st.caption("Frame-based feature extraction.")
    st.markdown("🔊 **Audio analysis**")
    st.caption("Audio feature extraction.")
    st.markdown("🧠 **Multimodal prediction**")
    st.caption("Combined model inference.")

    st.divider()
    st.markdown("### Model status")

    if checkpoint_files:
        st.success(f"{len(checkpoint_files)} checkpoint file(s) found")
    else:
        st.warning("No checkpoint file found")

    st.caption(
        "A checkpoint file being present does not guarantee "
        "that the model can make a valid prediction."
    )


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI-powered media research</div>
        <div class="hero-title">Deepfake Audio-Video Detection</div>
        <p class="hero-copy">
            Upload a video to analyze its visual and audio information
            using the project's multimodal deep learning model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# OVERVIEW CARDS
# --------------------------------------------------

a, b, c = st.columns(3)

with a:
    st.markdown(
        """
        <div class="status-card">
            <div class="status-label">Input</div>
            <div class="status-value">Video upload</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with b:
    st.markdown(
        """
        <div class="status-card">
            <div class="status-label">Maximum file size</div>
            <div class="status-value">200 MB</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">Model assets</div>
            <div class="status-value">
                {"Found" if checkpoint_files else "Not found"}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# VIDEO UPLOAD
# --------------------------------------------------

st.markdown(
    '<div class="section-title">01 · Upload video</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-copy">'
    'Supported formats: MP4, MOV, AVI, MKV and WEBM.'
    '</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "mov", "avi", "mkv", "webm"],
    help="Choose a video up to 200 MB.",
)


# --------------------------------------------------
# UPLOAD VALIDATION AND PREVIEW
# --------------------------------------------------

if uploaded_file is None:
    st.info("Choose a video file to see the preview and file information.")

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

        st.markdown(
            '<div class="section-copy">'
            'Review your uploaded video before starting the analysis.'
            '</div>',
            unsafe_allow_html=True,
        )

        left, center, right = st.columns([1, 2.8, 1])

        with center:
            st.video(uploaded_file)

        st.markdown("#### File information")

        x, y, z = st.columns(3)

        with x:
            st.metric("File size", f"{file_size_mb:.2f} MB")

        with y:
            st.metric("Format", file_format)

        with z:
            st.metric("Validation", "Passed")


        # --------------------------------------------------
        # RUN MODEL INFERENCE
        # --------------------------------------------------

        st.markdown(
            '<div class="section-title">03 · Detection</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-copy">'
            'Analyze the uploaded video using the trained multimodal model.'
            '</div>',
            unsafe_allow_html=True,
        )

        if not checkpoint_files:
            st.warning(
                "No checkpoint was found in the checkpoints folder. "
                "Prediction may fail until the correct checkpoint is available."
            )

        if st.button(
            "Run video analysis",
            type="primary",
            use_container_width=True,
        ):
            temp_path = None

            try:
                # Save the uploaded video temporarily so that the
                # inference pipeline can read it from a file path.
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(uploaded_file.name).suffix,
                ) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                with st.spinner(
                    "Analyzing video and audio. This may take a while..."
                ):
                    result = predict_video(temp_path)

                if not isinstance(result, dict):
                    raise ValueError(
                        "The prediction pipeline did not return a result dictionary."
                    )

                # Support the common result-key names used by the
                # inference pipeline.
                fake_score = result.get(
                    "fake_score",
                    result.get("average_fake_score"),
                )

                real_score = result.get("real_score")

                if fake_score is None:
                    raise KeyError(
                        "The inference result does not contain "
                        "'fake_score' or 'average_fake_score'."
                    )

                fake_score = float(fake_score)

                if real_score is None:
                    real_score = 1.0 - fake_score

                real_score = float(real_score)

                # Accept scores returned either as fractions (0–1)
                # or as percentages (0–100).
                if fake_score > 1.0:
                    fake_score /= 100.0

                if real_score > 1.0:
                    real_score /= 100.0

                if not (0.0 <= fake_score <= 1.0):
                    raise ValueError("Fake score is outside the valid range.")

                if not (0.0 <= real_score <= 1.0):
                    raise ValueError("Real score is outside the valid range.")

                prediction = result.get("prediction", result.get("label"))

                if prediction is None:
                    prediction = "FAKE" if fake_score >= 0.5 else "REAL"

                prediction = str(prediction).upper()

                # Display the prediction.
                st.markdown("### Detection results")

                if "FAKE" in prediction:
                    st.error(f"Prediction: {prediction}")
                elif "REAL" in prediction:
                    st.success(f"Prediction: {prediction}")
                else:
                    st.info(f"Prediction: {prediction}")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Prediction", prediction)

                with col2:
                    st.metric("Fake score", f"{fake_score * 100:.2f}%")

                with col3:
                    st.metric("Real score", f"{real_score * 100:.2f}%")

                # Additional model information, if returned.
                duration = result.get("duration")
                device = result.get("device")

                info_col1, info_col2 = st.columns(2)

                with info_col1:
                    if duration is not None:
                        st.metric("Video duration", f"{float(duration):.2f} sec")

                with info_col2:
                    if device is not None:
                        st.metric("Inference device", str(device))

                # Display individual-window results when available.
                windows = result.get(
                    "windows",
                    result.get("window_results", []),
                )

                if windows:
                    st.markdown("### Window-by-window analysis")

                    if isinstance(windows, list):
                        st.dataframe(
                            windows,
                            use_container_width=True,
                        )
                    else:
                        st.write(windows)

                warning = result.get("warning")

                if warning:
                    st.warning(str(warning))

                st.info(
                    "Research prototype: scores are model outputs, "
                    "not calibrated probabilities or proof that a video "
                    "is authentic or manipulated."
                )

            except Exception as exc:
                st.error(f"Video analysis failed: {exc}")

                st.markdown("**What to check:**")
                st.markdown(
                    "- Confirm that the model checkpoint exists.\n"
                    "- Confirm that the video contains an audio track.\n"
                    "- Confirm that the configuration and model files are available.\n"
                    "- Read the full error in the terminal running Streamlit."
                )

            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        AI Detection Lab · Research Prototype · Multimodal Deep Learning
    </div>
    """,
    unsafe_allow_html=True,
)
