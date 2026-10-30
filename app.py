from pathlib import Path
import os
import tempfile
import html as html_lib

import streamlit as st

from app_components.validation import validate_uploaded_file
from inference.predict_video import predict_video


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Deepfake Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent

CHECKPOINT_PATH = (
    BASE_DIR / "checkpoints" / "multimodal_reliability_best.pt"
)
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"

PAGES = [
    "Video Analysis",
    "Visual Analysis",
    "Audio Analysis",
    "Detection Results",
]

if st.session_state.get("navigation") not in PAGES:
    st.session_state["navigation"] = "Video Analysis"

if "theme" not in st.session_state:
    st.session_state["theme"] = "Light"


# ============================================================
# HELPERS
# ============================================================

def render_html(content):
    """Render HTML directly. Never render HTML as Markdown code."""
    st.html(content)


def go_to_video_analysis():
    st.session_state["navigation"] = "Video Analysis"


def get_result():
    return st.session_state.get("analysis_result")


def get_windows():
    result = get_result()
    if not result:
        return []

    windows = result.get(
        "windows",
        result.get("window_results", []),
    )

    return windows if isinstance(windows, list) else []


def get_score(value):
    if value is None:
        return None

    value = float(value)

    if 0 <= value <= 1:
        return value * 100

    return value


# ============================================================
# DESIGN SYSTEM
# Both themes are defined once here. The active theme is chosen
# by a marker element rendered on every run, so switching between
# Light and Dark never leaves old variables behind.
# ============================================================

dark = st.session_state["theme"] == "Dark"
theme_class = "theme-dark" if dark else "theme-light"

render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

/* ---------- Palettes ---------- */

.stApp {
    --bg: #f5f7ff;
    --bg-glow-1: rgba(79, 110, 247, .16);
    --bg-glow-2: rgba(124, 92, 255, .12);
    --surface: rgba(255, 255, 255, .82);
    --surface-solid: #ffffff;
    --surface-2: #f1f4ff;
    --text: #0e1a35;
    --muted: #5b6a86;
    --border: rgba(30, 50, 110, .10);
    --accent: #4f6ef7;
    --accent-2: #7c5cff;
    --accent-soft: rgba(79, 110, 247, .10);
    --green: #0f9d6b;
    --red: #e11d48;
    --shadow: 0 20px 50px -24px rgba(30, 50, 110, .28);
    --shadow-sm: 0 6px 18px -10px rgba(30, 50, 110, .18);
}

.stApp:has(.theme-dark) {
    --bg: #060a14;
    --bg-glow-1: rgba(124, 156, 255, .20);
    --bg-glow-2: rgba(165, 139, 255, .14);
    --surface: rgba(19, 28, 48, .72);
    --surface-solid: #131c30;
    --surface-2: #1a2540;
    --text: #eef2fb;
    --muted: #94a1bc;
    --border: rgba(148, 170, 220, .14);
    --accent: #7c9cff;
    --accent-2: #a58bff;
    --accent-soft: rgba(124, 156, 255, .14);
    --green: #34d399;
    --red: #fb7185;
    --shadow: 0 24px 60px -20px rgba(0, 0, 0, .65);
    --shadow-sm: 0 8px 22px -12px rgba(0, 0, 0, .5);
}

/* ---------- Base ---------- */

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased;
}

.stApp {
    background:
        radial-gradient(1100px 620px at 6% -12%, var(--bg-glow-1), transparent 60%),
        radial-gradient(900px 520px at 108% 2%, var(--bg-glow-2), transparent 55%),
        var(--bg);
    color: var(--text);
    min-height: 100vh;
}

#MainMenu, footer, [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stSidebar"],
[data-testid="collapsedControl"], [data-testid="stHeader"] {
    display: none !important;
}

.block-container {
    max-width: 1220px;
    padding: 0 2.5rem 4rem;
}

h1, h2, h3, h4, h5 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--text) !important;
    letter-spacing: -.4px;
    font-weight: 700 !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"] p {
    color: var(--muted);
}

[data-testid="stCaptionContainer"] {
    font-size: 12px;
}

/* ---------- Navbar ---------- */

.nav-shell {
    position: sticky;
    top: 14px;
    z-index: 999;
    margin: 18px 0 0;
    padding: 12px 14px 12px 18px;
    border-radius: 999px;
    background: var(--surface);
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
    backdrop-filter: saturate(180%) blur(18px);
    -webkit-backdrop-filter: saturate(180%) blur(18px);
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-mark {
    width: 38px;
    height: 38px;
    border-radius: 12px;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    display: grid;
    place-items: center;
    color: #fff;
    font-weight: 800;
    font-size: 17px;
    box-shadow: 0 10px 24px -8px var(--accent);
}

.brand-name {
    font-weight: 800;
    font-size: 15.5px;
    letter-spacing: -.3px;
    color: var(--text);
    line-height: 1.1;
}

.brand-tag {
    font-size: 10.5px;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: .4px;
    margin-top: 3px;
}

div[data-testid="stRadio"] > label {
    display: none;
}

div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    background: var(--surface-2);
    border: 1px solid var(--border);
    padding: 4px;
    border-radius: 999px;
    width: fit-content;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label {
    padding: 7px 14px;
    border-radius: 999px;
    cursor: pointer;
    margin: 0 !important;
    transition: background .2s ease, box-shadow .2s ease;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label p {
    font-size: 13px;
    font-weight: 600;
    color: var(--muted) !important;
    margin: 0;
    white-space: nowrap;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
    background: var(--surface-solid);
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    box-shadow: 0 8px 18px -8px var(--accent);
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) p {
    color: #fff !important;
}

[data-testid="stSelectbox"] label {
    display: none;
}

[data-testid="stSelectbox"] > div > div {
    background: var(--surface-solid);
    border: 1px solid var(--border);
    border-radius: 999px;
    min-height: 38px;
    color: var(--text);
}

.nav-divider {
    height: 1px;
    background: var(--border);
    margin: 22px 0 0;
}

/* ---------- Page header ---------- */

.page-head {
    padding: 48px 0 26px;
}

.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--accent);
    padding: 6px 12px;
    border-radius: 999px;
    background: var(--accent-soft);
    border: 1px solid var(--border);
}

.eyebrow::before {
    content: "";
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 0 4px rgba(52, 211, 153, .18);
}

.page-title {
    font-size: clamp(32px, 4vw, 46px);
    font-weight: 800;
    letter-spacing: -1.3px;
    line-height: 1.08;
    color: var(--text);
    margin: 16px 0 10px;
}

.page-description {
    font-size: 16px;
    line-height: 1.65;
    color: var(--muted);
    max-width: 640px;
}

/* ---------- Hero ---------- */

.hero {
    position: relative;
    overflow: hidden;
    display: grid;
    grid-template-columns: 1.15fr 1fr;
    gap: 40px;
    align-items: center;
    padding: 60px 56px;
    margin-top: 40px;
    border-radius: 30px;
    border: 1px solid var(--border);
    background: var(--surface);
    box-shadow: var(--shadow);
}

.hero::before {
    content: "";
    position: absolute;
    top: -180px;
    right: -120px;
    width: 520px;
    height: 520px;
    background: radial-gradient(circle, var(--accent-soft), transparent 65%);
    pointer-events: none;
}

.hero-chip {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    color: var(--accent);
    background: var(--accent-soft);
    border: 1px solid var(--border);
    padding: 6px 12px;
    border-radius: 999px;
}

.hero-title {
    font-size: clamp(36px, 4.4vw, 56px);
    font-weight: 800;
    letter-spacing: -1.8px;
    line-height: 1.04;
    color: var(--text);
    margin: 22px 0 18px;
}

.hero-title em {
    font-style: normal;
    background: linear-gradient(120deg, var(--accent), var(--accent-2));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.hero-description {
    font-size: 16.5px;
    line-height: 1.72;
    color: var(--muted);
    max-width: 520px;
}

.hero-actions {
    display: flex;
    gap: 12px;
    margin-top: 28px;
    flex-wrap: wrap;
}

.pill-ghost {
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
    padding: 11px 18px;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: var(--surface-solid);
}

/* Scanner visual */

.scanner {
    position: relative;
    aspect-ratio: 1 / 1;
    width: 100%;
    max-width: 380px;
    justify-self: center;
    border-radius: 26px;
    background:
        linear-gradient(var(--border) 1px, transparent 1px) 0 0 / 100% 30px,
        linear-gradient(90deg, var(--border) 1px, transparent 1px) 0 0 / 30px 100%,
        var(--surface-solid);
    border: 1px solid var(--border);
    overflow: hidden;
}

.scanner::after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    height: 96px;
    top: -96px;
    background: linear-gradient(180deg, transparent, var(--accent-soft) 70%, var(--accent));
    opacity: .8;
    animation: scan 3.4s ease-in-out infinite;
}

@keyframes scan {
    0%   { top: -96px; }
    100% { top: 100%; }
}

.scanner-core {
    position: absolute;
    inset: 22%;
    border-radius: 50%;
    border: 1.5px dashed var(--accent);
    display: grid;
    place-items: center;
    animation: spin 20s linear infinite;
}

.scanner-dot {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    box-shadow: 0 0 0 8px var(--accent-soft), 0 0 30px var(--accent);
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.scanner-tag {
    position: absolute;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px;
    color: var(--text);
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 6px 10px;
    border-radius: 9px;
    backdrop-filter: blur(8px);
}

/* ---------- Stat / feature cards ---------- */

.card-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
    margin: 26px 0 10px;
}

.card {
    position: relative;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 26px;
    box-shadow: var(--shadow-sm);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}

.card:hover {
    transform: translateY(-4px);
    border-color: var(--accent);
    box-shadow: var(--shadow);
}

.card-icon {
    width: 42px;
    height: 42px;
    border-radius: 13px;
    display: grid;
    place-items: center;
    background: var(--accent-soft);
    color: var(--accent);
    font-size: 18px;
    margin-bottom: 18px;
}

.card-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--muted);
}

.card-value {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -.6px;
    color: var(--text);
    margin-top: 8px;
}

.card-text {
    font-size: 14px;
    line-height: 1.6;
    color: var(--muted);
    margin-top: 6px;
}

/* Steps */

.steps {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
    margin-top: 10px;
}

.step {
    padding: 24px;
    border-radius: 22px;
    border: 1px solid var(--border);
    background: var(--surface);
}

.step-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    color: var(--accent);
}

.step-title {
    font-size: 17px;
    font-weight: 700;
    color: var(--text);
    margin: 10px 0 6px;
}

.step-text {
    font-size: 14px;
    line-height: 1.6;
    color: var(--muted);
}

/* ---------- Section headings ---------- */

.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--accent);
    margin: 40px 0 6px;
}

.section-title {
    font-size: 23px;
    font-weight: 800;
    letter-spacing: -.5px;
    color: var(--text);
    margin: 0 0 4px;
}

.section-desc {
    font-size: 14px;
    line-height: 1.6;
    color: var(--muted);
    margin-bottom: 18px;
}

/* ---------- Result ---------- */

.result {
    position: relative;
    overflow: hidden;
    padding: 32px 34px;
    border-radius: 26px;
    margin: 10px 0 24px;
    border: 1px solid var(--border);
    background: var(--surface);
    box-shadow: var(--shadow);
}

.result::after {
    content: "";
    position: absolute;
    right: -70px;
    top: -70px;
    width: 240px;
    height: 240px;
    border-radius: 50%;
    filter: blur(34px);
    opacity: .55;
}

.result-fake { border-color: rgba(225, 29, 72, .35); }
.result-fake::after { background: radial-gradient(circle, rgba(225, 29, 72, .5), transparent 70%); }

.result-real { border-color: rgba(15, 157, 107, .35); }
.result-real::after { background: radial-gradient(circle, rgba(15, 157, 107, .5), transparent 70%); }

.result-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    color: var(--muted);
}

.result-value {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1px;
    margin-top: 8px;
    color: var(--text);
}

.result-fake .result-value { color: var(--red); }
.result-real .result-value { color: var(--green); }

.result-description {
    font-size: 15px;
    color: var(--muted);
    margin-top: 6px;
    max-width: 560px;
}

/* Score bars */

.bar-wrap {
    margin-top: 22px;
    display: grid;
    gap: 14px;
}

.bar-row {
    display: grid;
    grid-template-columns: 110px 1fr 70px;
    align-items: center;
    gap: 12px;
    font-size: 13px;
    color: var(--muted);
    font-weight: 600;
}

.bar-track {
    height: 10px;
    border-radius: 999px;
    background: var(--surface-2);
    overflow: hidden;
    border: 1px solid var(--border);
}

.bar-fill {
    height: 100%;
    border-radius: 999px;
}

.bar-fill-fake { background: linear-gradient(90deg, #fb7185, var(--red)); }
.bar-fill-real { background: linear-gradient(90deg, #34d399, var(--green)); }

.bar-value {
    text-align: right;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text);
    font-size: 12.5px;
}

/* ---------- Streamlit component overrides ---------- */

.stButton > button {
    min-height: 50px;
    border-radius: 14px;
    border: none;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    color: #fff !important;
    font-weight: 700;
    font-size: 15px;
    box-shadow: 0 14px 30px -14px var(--accent);
    transition: transform .15s ease, box-shadow .2s ease, filter .2s ease;
}

.stButton > button p {
    color: #fff !important;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 18px 36px -14px var(--accent);
    filter: brightness(1.05);
}

.stButton > button:active {
    transform: translateY(0);
}

[data-testid="stFileUploader"] {
    background: var(--surface);
    border: 1.5px dashed var(--accent);
    border-radius: 22px;
    padding: 18px;
}

[data-testid="stFileUploader"] * {
    color: var(--text) !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: var(--surface-solid) !important;
    border-radius: 16px;
}

[data-testid="stVideo"] {
    border-radius: 22px;
    overflow: hidden;
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
}

[data-testid="stVerticalBlockBorderWrapper"] > div > div {
    background: var(--surface);
    border-radius: 22px;
}

[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 18px 20px;
    box-shadow: var(--shadow-sm);
}

[data-testid="stMetricLabel"] p {
    color: var(--muted) !important;
    font-size: 12.5px !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-weight: 800;
    letter-spacing: -.4px;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 20px;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}

[data-testid="stAlert"] {
    border-radius: 16px;
    border: 1px solid var(--border);
}

[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 18px;
    background: var(--surface);
}

hr {
    border-color: var(--border) !important;
    margin: 18px 0 !important;
}

/* ---------- Project progress ---------- */

.team-head {
    font-size: 17px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 8px;
}

/* ---------- Footer ---------- */

.site-footer {
    margin-top: 70px;
    padding: 28px 0 8px;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    font-size: 13px;
    color: var(--muted);
}

.site-footer .mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: .8px;
}

/* ---------- Responsive ---------- */

@media (max-width: 900px) {
    .block-container {
        padding: 0 1rem 2.5rem;
    }
    .hero {
        grid-template-columns: 1fr;
        padding: 34px 24px;
    }
    .scanner {
        max-width: 260px;
    }
    .card-grid,
    .steps {
        grid-template-columns: 1fr;
    }
    .nav-shell {
        border-radius: 22px;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        width: 100%;
        border-radius: 16px;
    }
    .bar-row {
        grid-template-columns: 90px 1fr 60px;
    }
}
</style>
""")

# Theme marker: selects the active palette above. Rendered on every run.
render_html(f'<div class="{theme_class}" style="display:none;"></div>')


# ============================================================
# TOP NAVBAR
# ============================================================

nav_brand, nav_links, nav_theme = st.columns([1.2, 3.2, 0.9], gap="small")

with nav_brand:
    render_html("""
    <div class="nav-shell" style="margin-top:18px;">
        <div class="brand">
            <div class="brand-mark">◈</div>
            <div>
                <div class="brand-name">Deepfake Intelligence</div>
                <div class="brand-tag">MEDIA FORENSICS · v1</div>
            </div>
        </div>
    </div>
    """)

with nav_links:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    selected_page = st.radio(
        "Navigation",
        PAGES,
        key="navigation",
        horizontal=True,
        label_visibility="collapsed",
    )

with nav_theme:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.selectbox(
        "Choose theme",
        ["Light", "Dark"],
        key="theme",
        label_visibility="collapsed",
    )

render_html('<div class="nav-divider"></div>')


# ============================================================
# PAGE HEADER
# ============================================================

PAGE_INFO = {
    "Video Analysis": (
        "MEDIA WORKSPACE",
        "Video Analysis",
        "Upload, preview, validate and analyze your video.",
    ),
    "Visual Analysis": (
        "VISUAL INTELLIGENCE",
        "Visual Analysis",
        "Inspect visual reliability information returned by the model.",
    ),
    "Audio Analysis": (
        "AUDIO INTELLIGENCE",
        "Audio Analysis",
        "Inspect audio reliability information returned by the model.",
    ),
    "Detection Results": (
        "DETECTION REPORT",
        "Detection Results",
        "Review the combined model prediction and its available evidence.",
    ),
}

eyebrow, title, description = PAGE_INFO[selected_page]

if selected_page in PAGES:
    render_html(f"""
    <div class="page-head">
        <div class="eyebrow">{eyebrow}</div>
        <div class="page-title">{title}</div>
        <div class="page-description">{description}</div>
    </div>
    """)


# ============================================================
# VIDEO ANALYSIS
# ============================================================

if selected_page == "Video Analysis":

    render_html("""
    <div class="section-label" style="margin-top:0;">Step 01</div>
    <div class="section-title">Upload video</div>
    <div class="section-desc">Supported formats: MP4, MOV, AVI, MKV and WEBM. Maximum size: 200 MB.</div>
    """)

    uploaded_file = st.file_uploader(
        "Choose a video",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        key="video_uploader",
    )

    if uploaded_file is None:

        with st.container(border=True):
            st.markdown("#### Your workspace is ready")
            st.write(
                "Select a video above to see its preview "
                "and start the detection process."
            )

    else:

        is_valid, validation_message = validate_uploaded_file(
            uploaded_file
        )

        file_key = f"{uploaded_file.name}:{uploaded_file.size}"

        if st.session_state.get("analysis_file_key") != file_key:
            st.session_state.pop("analysis_result", None)
            st.session_state["analysis_file_key"] = file_key

        if not is_valid:

            st.error(validation_message)
            st.session_state.pop("analysis_result", None)

        else:

            st.success("Video uploaded and validated successfully.")

            size_mb = uploaded_file.size / (1024 * 1024)
            extension = Path(uploaded_file.name).suffix.upper().lstrip(".")

            render_html("""
            <div class="section-label">Step 02</div>
            <div class="section-title">Video preview</div>
            """)

            preview_col, info_col = st.columns(
                [1.4, 1],
                gap="large",
            )

            with preview_col:
                st.video(uploaded_file.getvalue())

            with info_col:

                with st.container(border=True):

                    st.markdown("#### File information")

                    st.caption("FILE NAME")
                    st.write(uploaded_file.name)

                    st.divider()

                    st.metric("File size", f"{size_mb:.2f} MB")
                    st.metric("Format", extension)
                    st.metric("Validation", "Passed")

            render_html("""
            <div class="section-label">Step 03</div>
            <div class="section-title">Detection</div>
            """)

            if not CHECKPOINT_PATH.is_file():
                st.error(
                    f"Checkpoint missing: {CHECKPOINT_PATH}"
                )

            elif not CONFIG_PATH.is_file():
                st.error(
                    f"Configuration missing: {CONFIG_PATH}"
                )

            else:

                if st.button(
                    "✦ Run Multimodal Analysis",
                    type="primary",
                    use_container_width=True,
                ):

                    temp_path = None

                    try:

                        suffix = (
                            Path(uploaded_file.name).suffix.lower()
                            or ".mp4"
                        )

                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=suffix,
                        ) as temp_file:

                            temp_file.write(
                                uploaded_file.getvalue()
                            )
                            temp_path = temp_file.name

                        with st.spinner(
                            "Analyzing visual and audio information..."
                        ):

                            result = predict_video(temp_path)

                        if not isinstance(result, dict):
                            raise TypeError(
                                "Inference did not return a dictionary."
                            )

                        fake_score = get_score(
                            result.get("fake_score")
                        )

                        real_raw = result.get("real_score")

                        if fake_score is None:
                            raise KeyError(
                                "Inference result is missing fake_score."
                            )

                        real_score = (
                            get_score(real_raw)
                            if real_raw is not None
                            else 100.0 - fake_score
                        )

                        if not (
                            0 <= fake_score <= 100
                            and 0 <= real_score <= 100
                        ):
                            raise ValueError(
                                "Scores must be between 0 and 100."
                            )

                        prediction = str(
                            result.get("prediction", "")
                        ).strip().upper()

                        if prediction not in ("REAL", "FAKE"):
                            prediction = (
                                "FAKE" if fake_score >= 50 else "REAL"
                            )

                        st.session_state["analysis_result"] = {
                            **result,
                            "prediction": prediction,
                            "fake_score_percent": fake_score,
                            "real_score_percent": real_score,
                        }

                        st.success("Analysis completed successfully.")

                    except Exception as exc:

                        st.session_state.pop(
                            "analysis_result",
                            None,
                        )

                        st.error(f"Analysis failed: {exc}")

                        with st.expander("Technical error details"):
                            st.exception(exc)

                    finally:

                        if temp_path and os.path.exists(temp_path):
                            try:
                                os.remove(temp_path)
                            except OSError:
                                pass

            result = get_result()

            if result:

                render_html("""
                <div class="section-label">Step 04</div>
                <div class="section-title">Quick result</div>
                """)

                st.metric(
                    "Prediction",
                    result["prediction"],
                )

                st.write(
                    "Open the Detection Results page from the navigation "
                    "to see the full report."
                )


# ============================================================
# VISUAL ANALYSIS
# ============================================================

elif selected_page == "Visual Analysis":

    result = get_result()
    windows = get_windows()

    if not result:
        st.info(
            "No analysis is available yet. First upload a video "
            "and run analysis from the Video Analysis page."
        )

    else:

        render_html("""
        <div class="section-label" style="margin-top:0;">Reliability</div>
        <div class="section-title">Visual reliability by window</div>
        <div class="section-desc">Per-window visual scores as returned by the inference pipeline.</div>
        """)

        visual_rows = [
            {
                "Window": item.get("window"),
                "Start (sec)": item.get("start_seconds"),
                "End (sec)": item.get("end_seconds"),
                "Visual reliability": item.get("video_reliability"),
                "Prediction": item.get("prediction"),
            }
            for item in windows
            if item.get("video_reliability") is not None
        ]

        if visual_rows:

            st.dataframe(
                visual_rows,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Reliability values are shown as returned by the inference pipeline."
            )

        else:

            st.info(
                "Visual reliability values were not returned "
                "for this analysis."
            )


# ============================================================
# AUDIO ANALYSIS
# ============================================================

elif selected_page == "Audio Analysis":

    result = get_result()
    windows = get_windows()

    if not result:
        st.info(
            "No analysis is available yet. First upload a video "
            "and run analysis from the Video Analysis page."
        )

    else:

        render_html("""
        <div class="section-label" style="margin-top:0;">Reliability</div>
        <div class="section-title">Audio reliability by window</div>
        <div class="section-desc">Per-window audio scores as returned by the inference pipeline.</div>
        """)

        audio_rows = [
            {
                "Window": item.get("window"),
                "Start (sec)": item.get("start_seconds"),
                "End (sec)": item.get("end_seconds"),
                "Audio reliability": item.get("audio_reliability"),
                "Prediction": item.get("prediction"),
            }
            for item in windows
            if item.get("audio_reliability") is not None
        ]

        if audio_rows:

            st.dataframe(
                audio_rows,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Reliability values are shown as returned by the inference pipeline."
            )

        else:

            st.info(
                "Audio reliability values were not returned "
                "for this analysis."
            )


# ============================================================
# MULTIMODAL RESULTS
# ============================================================

elif selected_page == "Detection Results":

    result = get_result()

    if not result:

        st.info(
            "No detection report is available. Go to Video Analysis "
            "and run the model first."
        )

    else:

        prediction = result["prediction"]
        fake_score = result["fake_score_percent"]
        real_score = result["real_score_percent"]

        render_html("""
        <div class="section-label" style="margin-top:0;">Verdict</div>
        <div class="section-title">Model prediction</div>
        """)

        if prediction == "FAKE":

            render_html("""
            <div class="result result-fake">
                <div class="result-label">DETECTION RESULT</div>
                <div class="result-value">⚠ FAKE</div>
                <div class="result-description">
                    The model classified this video as potentially manipulated.
                </div>
            </div>
            """)

        else:

            render_html("""
            <div class="result result-real">
                <div class="result-label">DETECTION RESULT</div>
                <div class="result-value">✓ REAL</div>
                <div class="result-description">
                    The model classified this video as likely authentic.
                </div>
            </div>
            """)

        fake_w = max(0.0, min(100.0, float(fake_score)))
        real_w = max(0.0, min(100.0, float(real_score)))

        render_html(f"""
        <div class="bar-wrap">
            <div class="bar-row">
                <span>Fake score</span>
                <div class="bar-track"><div class="bar-fill bar-fill-fake" style="width:{fake_w:.2f}%;"></div></div>
                <span class="bar-value">{fake_w:.2f}%</span>
            </div>
            <div class="bar-row">
                <span>Real score</span>
                <div class="bar-track"><div class="bar-fill bar-fill-real" style="width:{real_w:.2f}%;"></div></div>
                <span class="bar-value">{real_w:.2f}%</span>
            </div>
        </div>
        """)

        m1, m2, m3 = st.columns(3, gap="medium")

        with m1:
            st.metric("Prediction", prediction)

        with m2:
            st.metric("Fake score", f"{fake_score:.2f}%")

        with m3:
            st.metric("Real score", f"{real_score:.2f}%")

        render_html("""
        <div class="section-label">Metadata</div>
        <div class="section-title">Additional information</div>
        """)

        duration = result.get(
            "duration_seconds",
            result.get("duration"),
        )

        device = result.get("device")

        d1, d2 = st.columns(2, gap="medium")

        with d1:
            if duration is not None:
                st.metric(
                    "Video duration",
                    f"{float(duration):.2f} seconds",
                )

        with d2:
            if device is not None:
                st.metric("Inference device", str(device))

        windows = get_windows()

        render_html("""
        <div class="section-label">Evidence</div>
        <div class="section-title">Window-wise evidence</div>
        """)

        if windows:

            st.dataframe(
                windows,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info("No window-wise evidence was returned.")

        if result.get("warning"):
            st.warning(str(result["warning"]))

        st.info(
            "Research prototype: model scores are not calibrated "
            "probabilities or proof that a video is authentic "
            "or manipulated."
        )


# ============================================================
# FOOTER
# ============================================================

render_html("""
<div class="site-footer">
    <span class="mono">DEEPFAKE INTELLIGENCE · MEDIA FORENSICS</span>
    <span>Multimodal Deep Learning Research Prototype</span>
</div>
""")