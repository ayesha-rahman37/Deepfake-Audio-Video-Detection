
from pathlib import Path

import streamlit as st
from app_components.validation import validate_uploaded_file

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

    /* Compact video preview with a maximum 280px height. */
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

    /* High contrast primary button. */
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

# Sidebar
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
        "Finding a checkpoint does not confirm that model inference is ready."
    )

# Header
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI-powered media research</div>
        <div class="hero-title">Deepfake Audio-Video Detection</div>
        <p class="hero-copy">
            Upload a video, review its preview and check the file details.
            Model-based detection will be available when the trained
            inference pipeline is connected.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Overview cards
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

# Upload section
st.markdown(
    '<div class="section-title">01 · Upload video</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-copy">Supported formats: MP4, MOV, AVI, MKV and WEBM.</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "mov", "avi", "mkv", "webm"],
    help="Choose a video up to 200 MB.",
)

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
            '<div class="section-copy">A compact player to review your uploaded video.</div>',
            unsafe_allow_html=True,
        )

        # Keep the preview centered and compact.
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

        st.markdown(
            '<div class="section-title">03 · Detection</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-copy">The model must be connected before a reliable prediction can be shown.</div>',
            unsafe_allow_html=True,
        )

        if checkpoint_files:
            st.warning(
                "Checkpoint files were found, but model loading, video/audio "
                "preprocessing and inference are not connected to this page."
            )
        else:
            st.warning(
                "No model checkpoint was found. A Real/Fake prediction "
                "cannot be generated yet."
            )

        if st.button(
            "Run video analysis",
            type="primary",
            use_container_width=True,
        ):
            st.error(
                "Analysis has not run: trained-model inference is not "
                "connected yet. No Real/Fake prediction was generated."
            )

st.markdown(
    """
    <div class="footer">
        AI Detection Lab · Research Prototype · Results require a connected
        trained inference pipeline.
    </div>
    """,
    unsafe_allow_html=True,
)
