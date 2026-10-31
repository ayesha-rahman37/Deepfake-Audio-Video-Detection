
import streamlit as st


def apply_theme():
    st.html("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        color-scheme: light;
        --bg: #f4f2f9;
        --surface: #ffffff;
        --surface-soft: #faf8fd;
        --surface-tint: #f0eafb;
        --text: #26213a;
        --muted: #777187;
        --border: #e8e2f0;
        --accent: #873be0;
        --accent-dark: #6e2bc0;
        --accent-soft: #f1e8fc;
        --green: #168768;
        --red: #d94c68;
        --shadow: 0 10px 32px rgba(48, 35, 75, .055);
        --shadow-hover: 0 16px 38px rgba(48, 35, 75, .10);
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 0% 0%, #eee5fa 0%, transparent 36%),
            radial-gradient(ellipse at 100% 5%, #e9edfb 0%, transparent 34%),
            var(--bg);
        color: var(--text);
    }

    .block-container {
        max-width: 1240px;
        padding: 1.4rem 2rem 3.5rem;
    }

    #MainMenu, footer, [data-testid="stDecoration"],
    [data-testid="stStatusWidget"], [data-testid="stHeader"],
    [data-testid="collapsedControl"], [data-testid="stSidebar"] {
        display: none !important;
    }

    h1, h2, h3, h4 {
        color: var(--text) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -.55px;
    }

    p, li {
        line-height: 1.7;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    /* Brand */

    .nav-brand {
        display: flex;
        align-items: center;
        gap: 13px;
        padding: 5px 0 14px;
    }

    .nav-logo {
        width: 48px;
        height: 48px;
        display: grid;
        place-items: center;
        flex-shrink: 0;
        border-radius: 15px;
        color: white;
        font-size: 23px;
        font-weight: 800;
        background: linear-gradient(135deg, #9638e8, #654de5);
        box-shadow: 0 8px 20px #9638e833;
    }

    .nav-name {
        font-size: 17px;
        font-weight: 800;
        color: var(--text);
        line-height: 1.3;
    }

    .nav-caption {
        font-size: 10px;
        color: var(--muted);
        letter-spacing: 1.2px;
        margin-top: 4px;
        font-weight: 600;
    }

    .nav-divider {
        border: 0;
        border-top: 1px solid var(--border);
        margin: 10px 0 22px;
    }

    /* Navigation buttons */

    div.stButton > button {
        min-height: 43px;
        width: 100%;
        border-radius: 11px;
        border: 1px solid #e7dcef;
        background: #ffffff;
        color: #554a69 !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        padding: 8px 10px;
        white-space: normal !important;
        line-height: 1.3 !important;
        box-shadow: 0 2px 7px rgba(40, 25, 60, .025);
        transition: .18s ease;
    }

    div.stButton > button p {
        color: inherit !important;
        font-size: inherit !important;
        font-weight: inherit !important;
        white-space: normal !important;
        line-height: 1.3 !important;
    }

    div.stButton > button:hover {
        border-color: #cba8ed;
        background: #f8f2ff;
        color: var(--accent-dark) !important;
        transform: translateY(-1px);
        box-shadow: 0 5px 13px rgba(100, 50, 150, .08);
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(120deg, #9638e8, #7049df);
        color: #ffffff !important;
        border: 1px solid transparent;
        box-shadow: 0 6px 15px #9638e82b;
    }

    div.stButton > button[kind="primary"] p {
        color: #ffffff !important;
    }

    div.stButton > button[kind="primary"]:hover {
        filter: brightness(1.05);
        color: #ffffff !important;
    }

    /* Hero */

    .hero {
        display: grid;
        grid-template-columns: 1.15fr .85fr;
        align-items: center;
        gap: 35px;
        padding: 55px 48px;
        border-radius: 26px;
        background: linear-gradient(125deg, #fff 0%, #f5effd 60%, #edf0fc 100%);
        border: 1px solid #e6dcef;
        box-shadow: var(--shadow);
        margin: 25px 0 38px;
        overflow: hidden;
    }

    .hero-kicker {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 30px;
        background: #f0e5fc;
        color: #812ecb;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.1px;
    }

    .hero-title {
        font-size: clamp(34px, 4.3vw, 54px);
        line-height: 1.13;
        font-weight: 800;
        letter-spacing: -1.8px;
        color: #252039;
        margin: 21px 0 15px;
    }

    .hero-title span {
        color: #9638e8;
    }

    .hero-description {
        font-size: 14px;
        color: #716b80;
        max-width: 530px;
        line-height: 1.8;
    }

    .hero-visual {
        min-height: 255px;
        border-radius: 22px;
        display: grid;
        place-items: center;
        background: radial-gradient(circle, #e8d7fb, #f4effb 55%, #e9edfb);
        border: 1px solid #e3d9f1;
    }

    .hero-orbit {
        width: 165px;
        height: 165px;
        border-radius: 50%;
        border: 2px dashed #b88ae6;
        display: grid;
        place-items: center;
    }

    .hero-core {
        width: 98px;
        height: 98px;
        border-radius: 29px;
        display: grid;
        place-items: center;
        font-size: 42px;
        color: white;
        background: linear-gradient(135deg, #9638e8, #654de5);
        box-shadow: 0 15px 35px #9638e855;
    }

    /* Headings */

    .section-kicker {
        color: #9638e8;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin: 28px 0 8px;
    }

    .section-title {
        font-size: 27px;
        font-weight: 800;
        color: var(--text);
        margin-bottom: 8px;
    }

    .section-description {
        color: var(--muted);
        font-size: 14px;
        margin-bottom: 22px;
    }

    /* Cards */

    .feature-card {
        height: 100%;
        min-height: 165px;
        padding: 24px;
        border-radius: 19px;
        background: #fff;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        transition: .2s ease;
    }

    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-hover);
        border-color: #d5b9ef;
    }

    .feature-icon {
        width: 45px;
        height: 45px;
        display: grid;
        place-items: center;
        border-radius: 14px;
        background: var(--accent-soft);
        font-size: 22px;
        margin-bottom: 16px;
    }

    .feature-title {
        font-size: 16px;
        font-weight: 800;
        color: var(--text);
        margin-bottom: 8px;
    }

    .feature-description {
        font-size: 13px;
        line-height: 1.7;
        color: var(--muted);
    }

    /* Authentication */

    .auth-heading {
        text-align: center;
        color: var(--text);
        font-size: 29px;
        font-weight: 800;
        margin: 10px 0 6px;
    }

    .auth-subtitle {
        text-align: center;
        color: var(--muted);
        font-size: 13px;
        margin-bottom: 24px;
    }

    [data-testid="stForm"] {
        padding: 25px;
        border: 1px solid var(--border);
        border-radius: 20px;
        background: #fff;
        box-shadow: var(--shadow);
    }

    [data-testid="stTextInput"] input,
    [data-testid="stPasswordInput"] input {
        border-radius: 11px;
        border: 1px solid #e2dce9;
        background: #fbfaff;
        color: var(--text);
    }

    /* Form buttons */

    .stFormSubmitButton > button {
        min-height: 46px;
        border-radius: 12px;
        border: none;
        background: linear-gradient(120deg, #9638e8, #654de5);
        color: white !important;
        font-weight: 700;
    }

    .stFormSubmitButton > button p {
        color: white !important;
    }

    /* Upload */

    [data-testid="stFileUploader"] {
        border: 1.5px dashed #bd91e5;
        border-radius: 18px;
        background: #faf7ff;
        padding: 15px;
    }

    [data-testid="stFileUploader"] * {
        color: var(--text) !important;
    }

    [data-testid="stVideo"] {
        border-radius: 17px;
        overflow: hidden;
    }

    /* Metrics and tables */

    [data-testid="stMetric"] {
        padding: 18px;
        border-radius: 17px;
        background: #fff;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
    }

    [data-testid="stMetricLabel"] p {
        color: var(--muted) !important;
        font-size: 12px;
    }

    [data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-weight: 800;
    }

    [data-testid="stDataFrame"] {
        border-radius: 14px;
        border: 1px solid var(--border);
        overflow: hidden;
    }

    [data-testid="stAlert"],
    [data-testid="stExpander"] {
        border-radius: 13px;
    }

    hr {
        border-color: var(--border) !important;
    }

    .site-footer {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
        padding-top: 23px;
        margin-top: 48px;
        border-top: 1px solid var(--border);
        color: var(--muted);
        font-size: 11px;
    }

    /* Responsive layout */

    @media(max-width: 850px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            grid-template-columns: 1fr;
            padding: 30px 24px;
        }

        .hero-visual {
            min-height: 200px;
        }

        .nav-name {
            font-size: 15px;
        }
    }
    </style>
    """)
