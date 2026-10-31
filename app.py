
from pathlib import Path
import os
import tempfile
import html

import pandas as pd
import streamlit as st

from app_components.auth import (
    init_db,
    create_user,
    authenticate_user,
    get_user_history,
    save_detection,
    get_detection_by_id,
)
from app_components.theme import apply_theme
from app_components.validation import validate_uploaded_file
from inference.predict_video import predict_video


# ============================================================
# 1. CONFIGURATION
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

init_db()
apply_theme()


# ============================================================
# 2. SESSION STATE
# ============================================================

DEFAULTS = {
    "user": None,
    "page": "Home",
    "analysis_result": None,
    "analysis_file_key": None,
    "last_saved_analysis_key": None,
    "selected_history_id": None,
    "registration_success": False,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


PUBLIC_PAGES = ["Home", "About", "Login", "Register"]

PRIVATE_PAGES = [
    "Dashboard",
    "Video Analysis",
    "Visual Analysis",
    "Audio Analysis",
    "Detection History",
    "Detection Results",
    "About",
    "Profile",
]


# ============================================================
# 3. NAVIGATION CALLBACKS
# ============================================================

def set_page(page):
    st.session_state["page"] = page


def logout():
    st.session_state["user"] = None
    st.session_state["analysis_result"] = None
    st.session_state["analysis_file_key"] = None
    st.session_state["last_saved_analysis_key"] = None
    st.session_state["selected_history_id"] = None
    st.session_state["page"] = "Home"


def safe(value):
    return html.escape(str(value))


def go_to(page):
    st.session_state["page"] = page
    st.rerun()


# ============================================================
# 4. GENERAL HELPERS
# ============================================================

def get_result():
    return st.session_state.get("analysis_result")


def get_windows(result=None):
    if result is None:
        result = get_result()

    if not result:
        return []

    windows = result.get("windows", [])
    return windows if isinstance(windows, list) else []


def get_score(value):
    if value is None:
        return None

    value = float(value)

    if 0 <= value <= 1:
        value *= 100

    return max(0.0, min(100.0, value))


def page_header(kicker, title, description):
    st.html(f"""
    <div style="padding:27px 0 17px">
        <div class="section-kicker">{safe(kicker)}</div>
        <div class="section-title">{safe(title)}</div>
        <div style="color:#777187;font-size:14px;margin-top:8px">
            {safe(description)}
        </div>
    </div>
    """)


def feature_card(icon, title, description):
    st.html(f"""
    <div class="feature-card">
        <div class="feature-icon">{icon}</div>
        <div class="feature-title">{safe(title)}</div>
        <div class="feature-description">
            {safe(description)}
        </div>
    </div>
    """)


def render_result_summary(result):
    prediction = result["prediction"]
    fake_score = result["fake_score"]
    real_score = result["real_score"]

    prediction_color = (
        "#d94c68" if prediction == "FAKE" else "#168768"
    )

    st.html(f"""
    <div class="feature-card" style="min-height:0;padding:27px">
        <div class="section-kicker">MODEL OUTPUT</div>
        <div style="
            font-size:36px;
            font-weight:800;
            color:{prediction_color};
            margin-top:10px;
        ">
            {safe(prediction)}
        </div>
        <div style="color:#777187;font-size:13px;margin-top:9px">
            Fake score: {fake_score:.2f}%
            &nbsp; | &nbsp;
            Real score: {real_score:.2f}%
        </div>
    </div>
    """)


def render_score_bars(fake_score, real_score):
    st.html(f"""
    <div style="
        background:#fff;
        border:1px solid #e8e2f0;
        border-radius:18px;
        padding:23px;
        box-shadow:0 8px 25px rgba(48,35,75,.04);
    ">
        <div style="
            font-weight:800;
            color:#26213a;
            margin-bottom:20px;
            font-size:16px;
        ">
            Prediction score overview
        </div>

        <div style="margin-bottom:20px">
            <div style="
                display:flex;
                justify-content:space-between;
                font-size:13px;
                margin-bottom:8px;
            ">
                <span style="color:#777187">Fake score</span>
                <b style="color:#d94c68">{fake_score:.2f}%</b>
            </div>
            <div style="
                height:10px;
                background:#f3e8ee;
                border-radius:20px;
                overflow:hidden;
            ">
                <div style="
                    height:100%;
                    width:{fake_score}%;
                    background:linear-gradient(90deg,#f28aa0,#d94c68);
                    border-radius:20px;
                "></div>
            </div>
        </div>

        <div>
            <div style="
                display:flex;
                justify-content:space-between;
                font-size:13px;
                margin-bottom:8px;
            ">
                <span style="color:#777187">Real score</span>
                <b style="color:#168768">{real_score:.2f}%</b>
            </div>
            <div style="
                height:10px;
                background:#e5f3ed;
                border-radius:20px;
                overflow:hidden;
            ">
                <div style="
                    height:100%;
                    width:{real_score}%;
                    background:linear-gradient(90deg,#58c9a1,#168768);
                    border-radius:20px;
                "></div>
            </div>
        </div>
    </div>
    """)


def render_window_table(windows):
    if not windows:
        st.info("No window-wise evidence is available.")
        return

    df = pd.DataFrame(windows)

    preferred = [
        "window",
        "start_seconds",
        "end_seconds",
        "fake_score",
        "prediction",
        "video_reliability",
        "audio_reliability",
    ]

    columns = [column for column in preferred if column in df.columns]

    if columns:
        df = df[columns]

    rename = {
        "window": "Window",
        "start_seconds": "Start (sec)",
        "end_seconds": "End (sec)",
        "fake_score": "Fake score (%)",
        "prediction": "Prediction",
        "video_reliability": "Visual reliability",
        "audio_reliability": "Audio reliability",
    }

    df = df.rename(columns=rename)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 5. TOP NAVIGATION
# ============================================================

user = st.session_state.get("user")

if user:
    navigation_items = [
        "Dashboard",
        "Video Analysis",
        "Visual Analysis",
        "Audio Analysis",
        "Detection History",
        "Detection Results",
        "About",
        "Profile",
    ]
else:
    navigation_items = ["Home", "About", "Login", "Register"]

if st.session_state["page"] not in navigation_items:
    st.session_state["page"] = "Dashboard" if user else "Home"


# Brand area
brand_col, account_col = st.columns([4, 1])

with brand_col:
    st.html("""
    <div class="nav-brand">
        <div class="nav-logo">◈</div>
        <div>
            <div class="nav-name">Deepfake Intelligence</div>
            <div class="nav-caption">MULTIMODAL MEDIA FORENSICS</div>
        </div>
    </div>
    """)

with account_col:
    if user:
        st.write("")
        st.button(
            "Logout",
            key="top_logout",
            use_container_width=True,
            on_click=logout,
        )

st.html('<hr class="nav-divider">')

# Navigation is arranged in two rows to preserve full labels.
for row_start in range(0, len(navigation_items), 4):

    row_items = navigation_items[row_start:row_start + 4]
    columns = st.columns(4, gap="small")

    for index, item in enumerate(row_items):

        with columns[index]:

            st.button(
                item,
                key=f"navigation_{item.lower().replace(' ', '_')}",
                type=(
                    "primary"
                    if st.session_state["page"] == item
                    else "secondary"
                ),
                use_container_width=True,
                on_click=set_page,
                args=(item,),
            )

st.divider()

current_page = st.session_state["page"]


# ============================================================
# 6. HOME PAGE
# ============================================================

if current_page == "Home":

    st.html("""
    <div class="hero">
        <div>
            <div class="hero-kicker">
                MULTIMODAL AI · DIGITAL MEDIA FORENSICS
            </div>

            <div class="hero-title">
                Discover the truth<br>
                behind every <span>video.</span>
            </div>

            <div class="hero-description">
                Analyze audio and visual evidence together
                with multimodal deep learning to identify
                potential deepfake manipulation.
            </div>
        </div>

        <div class="hero-visual">
            <div class="hero-orbit">
                <div class="hero-core">◈</div>
            </div>
        </div>
    </div>
    """)

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("### Media authenticity, made easier")

        st.write(
            "Deepfake Intelligence is a multimodal research "
            "platform that examines visual and audio signals "
            "together. Upload a video, run the model and "
            "review its prediction with supporting evidence."
        )

        if st.button("Get started", type="primary"):
            go_to("Register")

    with right:
        feature_card(
            "🧠",
            "Multimodal Intelligence",
            "A unified analysis workflow for visual evidence, "
            "audio evidence and combined prediction.",
        )

    st.html("""
    <div class="section-kicker">PLATFORM CAPABILITIES</div>
    <div class="section-title">Everything in one workspace</div>
    <div class="section-description">
        Explore the main analysis capabilities of the platform.
    </div>
    """)

    c1, c2, c3 = st.columns(3)

    with c1:
        feature_card(
            "🎥",
            "Visual Analysis",
            "Review visual reliability information across video windows.",
        )

    with c2:
        feature_card(
            "🔊",
            "Audio Analysis",
            "Explore audio reliability information from the model.",
        )

    with c3:
        feature_card(
            "📊",
            "Detection Results",
            "Understand model predictions, scores and supporting evidence.",
        )

    st.html("""
    <div class="section-kicker">HOW IT WORKS</div>
    <div class="section-title">A simple three-step process</div>
    """)

    a, b, c = st.columns(3)

    with a:
        feature_card(
            "01",
            "Upload",
            "Choose a supported video file for analysis.",
        )

    with b:
        feature_card(
            "02",
            "Analyze",
            "Run multimodal deep learning inference.",
        )

    with c:
        feature_card(
            "03",
            "Review",
            "Explore prediction scores and window-level evidence.",
        )

    st.write("")

    if st.button("Create your account", type="primary"):
        go_to("Register")


# ============================================================
# 7. ABOUT PAGE
# ============================================================

elif current_page == "About":

    page_header(
        "ABOUT THE PROJECT",
        "Deepfake Audio-Video Detection",
        "A multimodal deep learning research prototype "
        "for digital media analysis.",
    )

    st.write(
        "Deepfake Intelligence analyzes video frames and "
        "audio segments together to identify potentially "
        "manipulated media. The system produces an overall "
        "prediction and window-level reliability information."
    )

    a, b = st.columns(2)

    with a:
        feature_card(
            "🎥",
            "Visual Branch",
            "Processes sampled video frames and reports "
            "visual reliability for each analysis window.",
        )

    with b:
        feature_card(
            "🔊",
            "Audio Branch",
            "Processes audio segments and reports audio "
            "reliability for each analysis window.",
        )

    st.markdown("### Multimodal Fusion")

    st.write(
        "The model combines information from visual and "
        "audio branches through a reliability-based fusion "
        "mechanism to generate its prediction."
    )

    st.warning(
        "This is a research prototype. Its prediction is "
        "not proof that a video is authentic or manipulated."
    )


# ============================================================
# 8. REGISTER PAGE
# ============================================================

elif current_page == "Register":

    _, center, _ = st.columns([1, 1.5, 1])

    with center:

        st.html("""
        <div class="auth-heading">Create your account</div>
        <div class="auth-subtitle">
            Join Deepfake Intelligence and start analyzing videos.
        </div>
        """)

        with st.form("register_form"):

            full_name = st.text_input(
                "Full name",
                placeholder="Enter your full name",
            )

            email = st.text_input(
                "Email address",
                placeholder="name@example.com",
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="At least 8 characters",
            )

            confirm_password = st.text_input(
                "Confirm password",
                type="password",
            )

            submitted = st.form_submit_button(
                "Create Account",
                use_container_width=True,
            )

            if submitted:

                if password != confirm_password:
                    st.error("Passwords do not match.")

                else:
                    success, message = create_user(
                        full_name,
                        email,
                        password,
                    )

                    if success:
                        st.session_state["registration_success"] = True
                        go_to("Login")
                    else:
                        st.error(message)

        st.caption(
            "Already have an account? Select Login from the menu."
        )


# ============================================================
# 9. LOGIN PAGE
# ============================================================

elif current_page == "Login":

    _, center, _ = st.columns([1, 1.5, 1])

    with center:

        if st.session_state.get("registration_success"):
            st.success(
                "Your account has been created successfully. Please log in."
            )
            st.session_state["registration_success"] = False

        st.html("""
        <div class="auth-heading">Welcome back</div>
        <div class="auth-subtitle">
            Sign in to access your personal detection workspace.
        </div>
        """)

        with st.form("login_form"):

            email = st.text_input(
                "Email address",
                placeholder="name@example.com",
            )

            password = st.text_input(
                "Password",
                type="password",
            )

            submitted = st.form_submit_button(
                "Login",
                use_container_width=True,
            )

            if submitted:

                authenticated_user = authenticate_user(
                    email,
                    password,
                )

                if authenticated_user:

                    st.session_state["user"] = authenticated_user
                    st.session_state["analysis_result"] = None
                    st.session_state["analysis_file_key"] = None
                    st.session_state["last_saved_analysis_key"] = None
                    st.session_state["selected_history_id"] = None
                    st.session_state["page"] = "Dashboard"

                    st.rerun()

                else:
                    st.error("Invalid email or password.")

        st.caption(
            "New user? Select Register from the menu."
        )


# ============================================================
# 10. PRIVATE PAGES
# ============================================================

elif current_page in PRIVATE_PAGES:

    user = st.session_state.get("user")

    if not user:

        st.warning("Please log in to access your workspace.")

        if st.button("Go to Login"):
            go_to("Login")

    else:

        # ----------------------------------------------------
        # 10.1 DASHBOARD
        # ----------------------------------------------------

        if current_page == "Dashboard":

            history = get_user_history(user["id"])

            total = len(history)

            fake_count = sum(
                1 for row in history
                if row["prediction"] == "FAKE"
            )

            real_count = sum(
                1 for row in history
                if row["prediction"] == "REAL"
            )

            page_header(
                "YOUR WORKSPACE",
                f"Welcome back, {user['full_name']}",
                "Your personal overview of video analysis "
                "and detection activity.",
            )

            m1, m2, m3 = st.columns(3)

            m1.metric("Total Analyses", total)
            m2.metric("Potential Fake", fake_count)
            m3.metric("Potential Real", real_count)

            st.markdown("### Quick Actions")

            q1, q2 = st.columns(2)

            with q1:
                if st.button(
                    "🎬 Analyze a Video",
                    use_container_width=True,
                ):
                    go_to("Video Analysis")

            with q2:
                if st.button(
                    "📋 Detection History",
                    use_container_width=True,
                ):
                    go_to("Detection History")

            st.markdown("### Recent Activity")

            if history:
                st.dataframe(
                    pd.DataFrame(history[:5]),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(
                    "No analyses yet. Upload your first video to get started."
                )

        # ----------------------------------------------------
        # 10.2 VIDEO ANALYSIS
        # ----------------------------------------------------

        elif current_page == "Video Analysis":

            page_header(
                "MEDIA WORKSPACE",
                "Video Analysis",
                "Upload a video, validate it and run "
                "multimodal deepfake detection.",
            )

            uploaded_file = st.file_uploader(
                "Upload your video",
                type=["mp4", "mov", "avi", "mkv", "webm"],
                key="video_uploader",
                help=(
                    "Supported formats: MP4, MOV, AVI, MKV and WEBM. "
                    "Maximum file size: 200 MB."
                ),
            )

            if uploaded_file is None:

                st.info("Choose a video file to begin your analysis.")

            else:

                is_valid, validation_message = (
                    validate_uploaded_file(uploaded_file)
                )

                file_key = (
                    f"{user['id']}:"
                    f"{uploaded_file.name}:"
                    f"{uploaded_file.size}"
                )

                if st.session_state.get("analysis_file_key") != file_key:
                    st.session_state["analysis_file_key"] = file_key
                    st.session_state["analysis_result"] = None
                    st.session_state["last_saved_analysis_key"] = None

                if not is_valid:

                    st.error(validation_message)

                else:

                    st.success(validation_message)

                    preview_col, info_col = st.columns([1.4, 1])

                    with preview_col:
                        st.video(uploaded_file.getvalue())

                    with info_col:

                        st.markdown("### File Information")

                        st.write("**Filename:**", uploaded_file.name)

                        st.metric(
                            "File Size",
                            f"{uploaded_file.size / (1024 * 1024):.2f} MB",
                        )

                        st.metric(
                            "Format",
                            Path(uploaded_file.name).suffix.upper(),
                        )

                        st.metric("Validation", "Passed")

                    st.markdown("### Run Detection")

                    if not CHECKPOINT_PATH.is_file():

                        st.error(
                            f"Model checkpoint not found: {CHECKPOINT_PATH}"
                        )

                    elif not CONFIG_PATH.is_file():

                        st.error(
                            f"Configuration file not found: {CONFIG_PATH}"
                        )

                    else:

                        if st.button(
                            "Run Multimodal Analysis",
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

                                    temp_file.write(uploaded_file.getvalue())
                                    temp_path = temp_file.name

                                with st.spinner(
                                    "Analyzing visual and audio information..."
                                ):

                                    raw_result = predict_video(temp_path)

                                if not isinstance(raw_result, dict):
                                    raise TypeError(
                                        "Model output must be a dictionary."
                                    )

                                required_fields = [
                                    "prediction",
                                    "fake_score",
                                    "real_score",
                                    "windows",
                                ]

                                missing = [
                                    key for key in required_fields
                                    if key not in raw_result
                                ]

                                if missing:
                                    raise KeyError(
                                        "Missing model output fields: "
                                        + ", ".join(missing)
                                    )

                                prediction = str(
                                    raw_result["prediction"]
                                ).strip().upper()

                                if prediction not in ("REAL", "FAKE"):
                                    raise ValueError(
                                        "Model returned an invalid prediction."
                                    )

                                fake_score = get_score(
                                    raw_result["fake_score"]
                                )

                                real_score = get_score(
                                    raw_result["real_score"]
                                )

                                if (
                                    fake_score is None
                                    or real_score is None
                                ):
                                    raise ValueError(
                                        "Model returned missing scores."
                                    )

                                windows = raw_result["windows"]

                                if not isinstance(windows, list):
                                    raise ValueError(
                                        "Model returned invalid window evidence."
                                    )

                                final_result = {
                                    **raw_result,
                                    "prediction": prediction,
                                    "fake_score": fake_score,
                                    "real_score": real_score,
                                }

                                st.session_state[
                                    "analysis_result"
                                ] = final_result

                                if (
                                    st.session_state.get(
                                        "last_saved_analysis_key"
                                    ) != file_key
                                ):

                                    save_detection(
                                        user_id=user["id"],
                                        filename=uploaded_file.name,
                                        prediction=prediction,
                                        fake_score=fake_score,
                                        real_score=real_score,
                                        result=final_result,
                                    )

                                    st.session_state[
                                        "last_saved_analysis_key"
                                    ] = file_key

                                st.success(
                                    "Analysis completed and saved "
                                    "to your detection history."
                                )

                            except Exception as exc:

                                st.session_state["analysis_result"] = None

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

                        st.markdown("### Latest Prediction")

                        render_result_summary(result)

                        if st.button(
                            "View Full Detection Report",
                            use_container_width=True,
                        ):
                            go_to("Detection Results")

        # ----------------------------------------------------
        # 10.3 VISUAL ANALYSIS
        # ----------------------------------------------------

        elif current_page == "Visual Analysis":

            page_header(
                "MODEL EVIDENCE",
                "Visual Analysis",
                "Review visual reliability values returned "
                "for each video window.",
            )

            result = get_result()

            if not result:

                st.info(
                    "Run video analysis first. "
                    "The current session's result will appear here."
                )

            else:

                windows = get_windows(result)

                visual_rows = [
                    {
                        "Window": item.get("window"),
                        "Start (sec)": item.get("start_seconds"),
                        "End (sec)": item.get("end_seconds"),
                        "Visual Reliability": item.get(
                            "video_reliability"
                        ),
                        "Prediction": item.get("prediction"),
                    }
                    for item in windows
                ]

                if visual_rows:

                    st.dataframe(
                        visual_rows,
                        use_container_width=True,
                        hide_index=True,
                    )

                    values = [
                        float(item["video_reliability"])
                        for item in windows
                        if item.get("video_reliability") is not None
                    ]

                    if values:
                        st.metric(
                            "Average Visual Reliability",
                            f"{sum(values) / len(values):.4f}",
                        )

                else:
                    st.info(
                        "No visual evidence was returned by the model."
                    )

        # ----------------------------------------------------
        # 10.4 AUDIO ANALYSIS
        # ----------------------------------------------------

        elif current_page == "Audio Analysis":

            page_header(
                "MODEL EVIDENCE",
                "Audio Analysis",
                "Review audio reliability values returned "
                "for each video window.",
            )

            result = get_result()

            if not result:

                st.info(
                    "Run video analysis first. "
                    "The current session's result will appear here."
                )

            else:

                windows = get_windows(result)

                audio_rows = [
                    {
                        "Window": item.get("window"),
                        "Start (sec)": item.get("start_seconds"),
                        "End (sec)": item.get("end_seconds"),
                        "Audio Reliability": item.get(
                            "audio_reliability"
                        ),
                        "Prediction": item.get("prediction"),
                    }
                    for item in windows
                ]

                if audio_rows:

                    st.dataframe(
                        audio_rows,
                        use_container_width=True,
                        hide_index=True,
                    )

                    values = [
                        float(item["audio_reliability"])
                        for item in windows
                        if item.get("audio_reliability") is not None
                    ]

                    if values:
                        st.metric(
                            "Average Audio Reliability",
                            f"{sum(values) / len(values):.4f}",
                        )

                else:
                    st.info(
                        "No audio evidence was returned by the model."
                    )

        # ----------------------------------------------------
        # 10.5 DETECTION HISTORY
        # ----------------------------------------------------

        elif current_page == "Detection History":

            page_header(
                "YOUR RECORDS",
                "Detection History",
                "Review analyses saved under your account.",
            )

            history = get_user_history(
                user["id"],
                limit=200,
            )

            if not history:

                st.info("You have not analyzed any videos yet.")

            else:

                history_df = pd.DataFrame(history)

                st.dataframe(
                    history_df,
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("### Open a Saved Report")

                options = {
                    (
                        f"#{row['id']} · "
                        f"{row['filename']} · "
                        f"{row['prediction']}"
                    ): row["id"]
                    for row in history
                }

                selected_label = st.selectbox(
                    "Select an analysis",
                    list(options.keys()),
                )

                if st.button(
                    "View Saved Report",
                    type="primary",
                    use_container_width=True,
                ):

                    st.session_state[
                        "selected_history_id"
                    ] = options[selected_label]

                    go_to("Detection Results")

        # ----------------------------------------------------
        # 10.6 DETECTION RESULTS
        # ----------------------------------------------------

        elif current_page == "Detection Results":

            page_header(
                "ANALYSIS REPORT",
                "Detection Results",
                "Review the model prediction, scores and "
                "window-wise evidence.",
            )

            result = get_result()

            selected_id = st.session_state.get("selected_history_id")

            if not result and selected_id:

                saved = get_detection_by_id(
                    user["id"],
                    selected_id,
                )

                if saved:
                    result = saved.get("result", {})
                else:
                    st.warning(
                        "The selected report could not be found."
                    )

            if not result:

                st.info(
                    "No report available. Analyze a video or "
                    "select a saved report from Detection History."
                )

                if st.button("Go to Video Analysis"):
                    go_to("Video Analysis")

            else:

                prediction = str(
                    result.get("prediction", "UNKNOWN")
                ).upper()

                fake_score = get_score(
                    result.get(
                        "fake_score",
                        result.get("fake_score_percent"),
                    )
                )

                real_score = get_score(
                    result.get(
                        "real_score",
                        result.get("real_score_percent"),
                    )
                )

                if fake_score is None or real_score is None:

                    st.error(
                        "This report does not contain valid prediction scores."
                    )

                else:

                    render_result_summary({
                        "prediction": prediction,
                        "fake_score": fake_score,
                        "real_score": real_score,
                    })

                    render_score_bars(fake_score, real_score)

                    st.info(
                        "These percentages are model output scores. "
                        "They should not be interpreted as calibrated "
                        "probabilities or definitive proof of manipulation."
                    )

                    st.markdown("### Video Information")

                    m1, m2, m3 = st.columns(3)

                    duration = result.get("duration_seconds")
                    device = result.get("device")
                    windows = result.get("windows", [])

                    m1.metric(
                        "Video Duration",
                        (
                            f"{float(duration):.2f} sec"
                            if duration is not None
                            else "N/A"
                        ),
                    )

                    m2.metric(
                        "Processing Device",
                        str(device) if device else "N/A",
                    )

                    m3.metric(
                        "Analyzed Windows",
                        len(windows) if isinstance(windows, list) else 0,
                    )

                    st.markdown("### Window-wise Evidence")

                    render_window_table(windows)

                    warning = result.get("warning")

                    if warning:
                        st.warning(str(warning))

                    st.caption(
                        "Research prototype: always review the source media "
                        "and supporting evidence before drawing a conclusion."
                    )

        # ----------------------------------------------------
        # 10.7 PROFILE
        # ----------------------------------------------------

        elif current_page == "Profile":

            page_header(
                "ACCOUNT",
                "My Profile",
                "View your account information.",
            )

            feature_card(
                "👤",
                user["full_name"],
                user["email"],
            )

            st.markdown("### Account Details")

            st.write("**Account ID:**", user["id"])
            st.write("**Full Name:**", user["full_name"])
            st.write("**Email:**", user["email"])
            st.write("**Member Since:**", user["created_at"])

            st.markdown("### Account Actions")

            if st.button(
                "Logout",
                type="primary",
                use_container_width=True,
            ):
                logout()
                st.rerun()


# ============================================================
# 11. FOOTER
# ============================================================

st.html("""
<div class="site-footer">
    <span>
        <b>DEEPFAKE INTELLIGENCE</b>
        · MEDIA FORENSICS
    </span>

    <span>
        Multimodal Deep Learning Research Prototype
    </span>
</div>
""")
