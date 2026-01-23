import streamlit as st
st.set_page_config(
    page_title="InstruNet AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import numpy as np
import librosa.display
import sys
from pathlib import Path
import tempfile
import librosa
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.auth import (
    create_users_table,
    register_user,
    authenticate_user
)


create_users_table()

from backend.pipeline import run_pipeline
from backend.export import export_json, export_pdf

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "home"
if "result" not in st.session_state:
    st.session_state.result = None
    
# ---------------- STYLING ----------------
st.markdown("""
<style>

/* ================= COLOR PALETTE ================= */
:root {
    --bg-main: linear-gradient(135deg, #020617 0%, #0f172a 50%, #1e293b 100%);
    --bg-sidebar: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    --bg-card: rgba(255,255,255,0.95);
    --white: #ffffff;
    --white-soft: rgba(255,255,255,0.85);
    --white-muted: rgba(255,255,255,0.65);
    --dark: #111827;
    --primary: #3b82f6;
}

/* ================= FULL BACKGROUND FIX ================= */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-main) !important;
}

header {
    background: transparent !important;
}

/* ================= SIDEBAR ================= */
[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
}

/* ================= GLOBAL TEXT ================= */
h1, h2, h3, h4, h5, h6,
p, span, label, small,
.stMarkdown, .stText, .stCaption {
    color: white !important;
}

/* ================= INPUT LABELS ================= */
[data-testid="stTextInput"] label,
[data-testid="stTextInput"] span {
    color: white !important;
}

/* ================= INPUT BOX ================= */
input, textarea {
    background: white !important;
    color: var(--dark) !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 12px !important;
}

input:focus, textarea:focus {
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.45) !important;
}

/* ================= FILE UPLOADER ================= */
[data-testid="stFileUploader"] {
    background: white !important;
    border-radius: 16px !important;
    padding: 18px !important;
    border: none !important;
    outline: none !important;
    box-shadow: 0 8px 20px rgba(0,0,0,0.25) !important;
}

section[data-testid="stSidebar"] [data-testid="stFileUploader"] * {
    color: #111827 !important;
}

/* ================= POPUP MESSAGES ================= */

/* Base styles for all alerts */
.stAlert,
.stSuccess,
.stError,
.stWarning,
.stInfo {
    background: rgba(0,0,0,0.45) !important; /* semi-transparent dark bg */
    border-radius: 10px !important;          /* slightly rounded corners */
    color: white !important;                 /* text color white */
    padding: 6px 12px !important;            /* small padding */
    font-size: 0.95rem !important;           /* slightly smaller text */
    min-height: auto !important;
    line-height: 1.3 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4) !important; /* subtle shadow */
}

/* Make all child elements inherit white text */
.stAlert *, 
.stSuccess *, 
.stInfo *, 
.stWarning *, 
.stError * {
    color: white !important;
}

/* Toast messages */
[data-testid="stToast"], 
[data-testid="stToast"] * {
    background: rgba(0,0,0,0.45) !important; /* dark semi-transparent bg */
    color: white !important;                 /* white text */
    padding: 6px 12px !important;
    font-size: 0.95rem !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4) !important;
}


/* ================= BUTTONS ================= */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
    color: white !important;
    border-radius: 14px !important;
    border: none !important;
}

/* ================= AUDIO ================= */
[data-testid="stAudio"] {
    border-radius: 14px !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.45) !important;
}

/* ================= EXPANDERS ================= */
.stExpanderHeader {
    background: linear-gradient(135deg, #1e293b, #334155) !important;
    color: white !important;
}

.stExpanderContent {
    background: white !important;
    color: var(--dark) !important;
}

/* ================= LAYOUT FIX ================= */
.block-container {
    padding-top: 2rem !important;
}
            
/* ================= CENTER ALIGN BUTTONS ================= */
.stButton {
    display: flex !important;
    justify-content: center !important;
}

.stButton > button {
    min-width: 160px !important;
}

/* ================= DASHBOARD BUTTON TEXT FIX ================= */
.stButton > button span {
    color: white !important;
}

/* ================= SLIDER TEXT VISIBILITY ================= */
[data-testid="stSlider"] * {
    color: white !important;
}

/* Slider value (0.30) */
[data-testid="stSlider"] .stSliderValue {
    color: white !important;
    font-weight: 600 !important;
}

/* Slider min/max labels */
[data-testid="stSlider"] .stMarkdown {
    color: white !important;
}

/* ================= REMOVE SLIDER OUTLINE ================= */
[data-testid="stSlider"] input {
    outline: none !important;
    box-shadow: none !important;
}                                    

/* ================= DOWNLOAD BUTTON TEXT FIX ================= */
[data-testid="stDownloadButton"] button {
    color: white !important;
}

[data-testid="stDownloadButton"] button span {
    color: white !important;
    font-weight: 600 !important;
}

/* Optional: improve button look */
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
    border-radius: 14px !important;
    padding: 12px 28px !important;
    border: none !important;
}
/* Make Streamlit expander headers visible in dark theme */
.stExpanderHeader {
    background: #1e293b !important; /* dark-blue header */
    color: white !important;        /* text visible */
    border: 1px solid #3b82f6 !important; /* optional border */
    border-radius: 8px !important;
    padding: 6px 12px !important;
}

/* Change arrow color */
.stExpanderHeader svg {
    stroke: white !important;
}

/* ================= MOBILE RESPONSIVENESS ================= */

/* Phones */
@media (max-width: 768px) {

    h1 {
        font-size: 1.6rem !important;
        text-align: center !important;
    }

    h2 {
        font-size: 1.3rem !important;
    }

    h3 {
        font-size: 1.1rem !important;
    }

    .block-container {
        padding: 1rem !important;
    }

    /* Buttons full width */
    .stButton > button {
        width: 100% !important;
        font-size: 1rem !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        padding: 12px !important;
    }

    /* Charts scale nicely */
    canvas {
        max-width: 100% !important;
        height: auto !important;
    }
}

/* Tablets */
@media (min-width: 769px) and (max-width: 1024px) {

    h1 {
        font-size: 2rem !important;
    }

    .block-container {
        padding: 1.5rem !important;
    }
}
            
/* ================= EXPANDER VISIBILITY FIX ================= */

/* Expander header base */
div[data-testid="stExpander"] > details > summary {
    background: linear-gradient(135deg, #1e293b, #334155) !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    border-left: 4px solid #3b82f6 !important; /* blue accent */
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.25s ease-in-out !important;
}

/* Hover effect (VERY IMPORTANT) */
div[data-testid="stExpander"] > details > summary:hover {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(0,0,0,0.45) !important;
}

/* When expanded */
div[data-testid="stExpander"] > details[open] > summary {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
}

/* Expander arrow */
div[data-testid="stExpander"] summary svg {
    stroke: white !important;
    transform: scale(1.2);
}

/* Content area */
div[data-testid="stExpander"] > details > div {
    background: rgba(255,255,255,0.95) !important;
    color: #111827 !important;
    border-radius: 0 0 12px 12px !important;
    padding: 14px !important;
}

/* Text inside expander content */
div[data-testid="stExpander"] > details > div * {
    color: #111827 !important;
}

                                
</style>
""", unsafe_allow_html=True)


# ---------------- STYLING ----------------

# ---------------- HOME PAGE ----------------
def home_page():
    container = st.container()
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<h1 style='text-align:center; color:#1E90FF'>🎵 InstruNet AI</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center; color:#333'>Professional AI-Based Music Instrument Recognition</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#555'>Empower your audio analysis with AI-powered multi-instrument detection.<br>Upload audio, analyze segments, and get detailed confidence reports.</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Explore", key="explore"):
            st.session_state.page = "login"
            st.rerun()

#------login page--------
def login_page():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.subheader("🔐 Login")

        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.user = username
                st.session_state.page = "app"
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

        st.divider()

        if st.button("📝 New user? Register"):
            st.session_state.page = "register"
            st.rerun()

#----Register-----
def register_page():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.subheader("📝 Register")

        new_user = st.text_input("New Username", key="reg_user")
        new_pass = st.text_input("New Password", type="password", key="reg_pass")

        if st.button("Register"):
            if register_user(new_user, new_pass):
                st.success("User registered successfully!")
                st.session_state.page = "login"
                st.rerun()
            else:
                st.error("Username already exists")

        st.divider()

        if st.button("🔐 Already have an account? Login"):
            st.session_state.page = "login"
            st.rerun()

# ---------------- DASHBOARD ----------------
def main_app():
    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.markdown(
            f"""
            <div style='text-align:center; margin-bottom:20px;'>
                <div style='width:70px;height:70px;background-color:#1E90FF;
                border-radius:50%;display:flex;justify-content:center;align-items:center;
                font-size:28px;color:white;margin:0 auto;'>
                    {st.session_state.user[0].upper()}
                </div>
                <p style='text-align:center;margin-top:5px;font-weight:bold;'>
                    {st.session_state.user}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.header("Controls")
        audio = st.file_uploader("Upload audio", ["wav", "mp3"])
        threshold = st.slider("Detection threshold", 0.1, 0.9, 0.3)
        aggregation = st.selectbox("Aggregation", ["mean", "max", "topk_mean"])

        run = st.button("▶ Predict",disabled=audio is None)

        st.markdown("<br><br>", unsafe_allow_html=True)

        if st.button("↩️ Logout"):
            st.session_state.logged_in = False
            st.session_state.page = "home"
            st.rerun()

    # ---------------- MAIN CONTENT ----------------
    st.title("🎵 InstruNet AI")

    if audio is None:
        st.info("Upload an audio file to begin.")
        return

    st.success(f"🎧 Audio loaded: **{audio.name}**")
    st.audio(audio)

    # ---------------- RUN MODEL ----------------
    if run:
        with st.spinner("Running inference..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio.read())
                st.session_state.audio_path = tmp.name

            st.session_state.result = run_pipeline(
                audio_path=st.session_state.audio_path,
                original_filename=audio.name,
                threshold=threshold,
                aggregation=aggregation
            )

        st.success("✅ Analysis completed")

    result = st.session_state.get("result")
    if result is None:
        st.info("Click Predict to analyze.")
        return

    # ---------------- DETECTED INSTRUMENTS ----------------
    st.markdown("### 🎯 Detected Instruments")
    final = result["predictions"]["final_instruments"]

    if not final:
        st.warning("No dominant instruments detected.")
    else:
        for inst in final:
            st.success(inst.upper())

    st.divider()

    # ---------------- CONFIDENCE SCORES ----------------
    st.subheader("📊 Confidence Scores")
    with st.expander("📊 Click to view Confidence Scores"):
        scores = result["predictions"]["confidence_scores"]
        instruments = list(scores.keys())
        values = list(scores.values())

        fig, ax = plt.subplots(figsize=(5.5, 2.2), dpi=100)
        ax.barh(instruments, values)

        for i, v in enumerate(values):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center")

        ax.set_xlim(0, 1)
        ax.set_xlabel("Confidence")
        st.pyplot(fig)

    # ---------------- AUDIO WAVEFORM ----------------
    st.subheader("🔊 Audio Waveform")
    with st.expander("🔊 Click to view Audio Waveform"):
        y, sr = librosa.load(st.session_state.audio_path, sr=16000)

        fig, ax = plt.subplots(figsize=(8.5, 2.0), dpi=100)
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

    # ---------------- MEL SPECTROGRAM ----------------
    st.subheader("🎧 Mel Spectrogram")
    with st.expander("🎧 Click to view Mel Spectrogram"):
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=64
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        fig, ax = plt.subplots(figsize=(7.5, 2.2), dpi=100)
        img = librosa.display.specshow(
            mel_db,
            sr=sr,
            x_axis="time",
            y_axis="mel",
            ax=ax
        )
        fig.colorbar(img, ax=ax)
        st.pyplot(fig)

    # ---------------- TIMELINE ----------------
    st.subheader("🕒 Instrument Confidence Timeline")
    with st.expander("🕒 Click to view Confidence Timeline"):
        timelines = np.array(result["predictions"]["timelines"])
        labels = list(result["predictions"]["confidence_scores"].keys())
        t = np.arange(timelines.shape[0])

        fig, ax = plt.subplots(figsize=(8.5, 2.6), dpi=100)
        for i, label in enumerate(labels):
            ax.plot(t, timelines[:, i], label=label)

        ax.set_ylim(0, 1)
        ax.set_xlabel("Time Segments (2s)")
        ax.set_ylabel("Confidence")
        ax.legend(ncol=4, fontsize=8)
        st.pyplot(fig)

    # ---------------- EXPORT ----------------
    st.markdown("### 📤 Export")
    st.download_button(
        "Download JSON",
        export_json(result),
        "instrunet_result.json"
    )
    st.download_button(
        "Download PDF",
        export_pdf(result),
        "instrunet_result.pdf"
    )

#-----Page router-----    

if st.session_state.page == "home":
    home_page()

elif st.session_state.page == "login":
    login_page()

elif st.session_state.page == "register":
    register_page()

elif st.session_state.page == "app":
    if st.session_state.logged_in:
        main_app()
    else:
        st.session_state.page = "login"
        st.rerun()
