import numpy as np
import librosa.display
import sys
from pathlib import Path
import tempfile
import streamlit as st
import librosa
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from backend.pipeline import run_pipeline
from backend.export import export_json, export_pdf

st.set_page_config(
    page_title="InstruNet AI",
    layout="wide"
)

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
/* ================= POPUP MESSAGES - DARK THEME OPTIMIZED ================= */

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

# ---------------- LOGIN PAGE ----------------
def login_page():
    container = st.container()
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<h1 style='text-align:center; color:#6C63FF'>🎵 InstruNet AI</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center; color:#333'>CNN-Based Music Instrument Recognition System</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#555'>Upload audio • Analyze • Detect instruments</p>", unsafe_allow_html=True)
        st.markdown("---")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", key="login"):
            if username.strip() and password.strip():
                st.session_state.logged_in = True
                st.session_state.user = username
                st.session_state.page = "app"
                st.success("✅Logged in successfully!")

                st.rerun()
            else:
                st.error("Username and password cannot be empty")

# ---------------- DASHBOARD ----------------
def main_app():
    with st.sidebar:
        # User info
        st.markdown(f"<div style='text-align:center; margin-bottom:20px;'>"
                    f"<div style='width:70px;height:70px;background-color:#1E90FF;"
                    f"border-radius:50%;display:flex;justify-content:center;align-items:center;"
                    f"font-size:28px;color:white;margin:0 auto;'>{st.session_state.user[0].upper()}</div>"
                    f"<p style='text-align:center;margin-top:5px;font-weight:bold;'>{st.session_state.user}</p>"
                    f"</div>", unsafe_allow_html=True)

        st.header("Controls")
        audio = st.file_uploader("Upload audio", ["wav","mp3"])
        threshold = st.slider("Detection threshold", 0.1, 0.9, 0.3)
        aggregation = st.selectbox("Aggregation", ["mean","max","topk_mean"])
        if st.button("▶ Predict", key="predict"):
            run = True
        else:
            run = False

        st.markdown("<br><br><br>", unsafe_allow_html=True)
        if st.button("↩️ Logout", key="logout"):
            st.success("Logged out successfully!")
            st.session_state.logged_in = False
            st.session_state.page = "home"
            st.rerun()

    # Main content
    st.title("🎵 InstruNet AI")
    if audio is None:
        st.info("Upload an audio file to begin.")
        st.stop()

    st.success(f"🎧 Audio loaded successfully: **{audio.name}**")
    st.audio(audio)

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
            st.success("✅ Analysis completed successfully!")

    result = st.session_state.get("result")
    if result is None:
        st.info("Click Predict to analyze.")
        st.stop()

    # ---------------- DETECTED INSTRUMENTS ----------------
    st.markdown("### 🎯 Detected Instruments")
    final = result["predictions"]["final_instruments"]
    if not final:
        st.warning("No dominant instruments detected.")
    else:
        for inst in final:
            st.success(inst.upper())

    st.info("💡 Click on the respective sections below to view detailed results.")
    st.divider()
    
    # ---------------- CONFIDENCE SCORES ----------------
    st.subheader("📊 Confidence Scores")
    with st.expander("Confidence scores"):
        scores = result["predictions"]["confidence_scores"]
        instruments = list(scores.keys())
        values = list(scores.values())

        fig, ax = plt.subplots(figsize=(6, 3))  # smaller fixed height
        bars = ax.barh(instruments, values, color="#3b82f6")  # blue bars

        for i, v in enumerate(values):
            ax.text(v + 0.01, i, f"{v:.2f}", color='white', va='center', fontweight='bold')

        ax.set_xlim(0,1)
        ax.set_xlabel("Confidence",color='white')
        ax.set_title("Instrument Confidence Scores", color='white')
        ax.set_facecolor("#0f172a")
        fig.patch.set_facecolor("#0f172a")  # match dark theme
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')

        st.pyplot(fig)
    st.divider()

    # ---------------- MEL SPECTROGRAM ----------------
    st.subheader("🎧 Mel Spectrogram")    
    with st.expander("🎧 Mel-Spectrogram"):
        if "audio_path" not in st.session_state:
           st.warning("Please run prediction first.")
           st.stop()

        y, sr = librosa.load(st.session_state.audio_path, sr=16000)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        fig, ax = plt.subplots(figsize=(8, 3))
        img = librosa.display.specshow(
            mel_db,
            sr=sr,
            x_axis="time",
            y_axis="mel",
            cmap="magma",
            ax=ax
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        st.pyplot(fig)    
    st.divider()

    # ---------------- TIMELINES ----------------
    st.subheader("🕒 Instrument Confidence Timeline")
    with st.expander("🕒 Instrument Confidence Timeline"):
        timelines = np.array(result["predictions"]["timelines"])
        labels = list(result["predictions"]["confidence_scores"].keys())

        time_axis = np.arange(timelines.shape[0])

        fig, ax = plt.subplots(figsize=(10, 4))
        for i, label in enumerate(labels):
            ax.plot(
                time_axis,
                timelines[:, i],
                label=label,
                linewidth=2
            )

        ax.set_xlabel("Time Segments (2s each)")
        ax.set_ylabel("Confidence")
        ax.set_ylim(0, 1)
        ax.legend(ncol=3, fontsize=8)
        ax.grid(alpha=0.3)

        st.pyplot(fig)
    st.divider()

    # ---------------- EXPORT ----------------
    st.markdown("### 📤 Export")
    st.download_button("Download JSON", export_json(result), "instrunet_result.json")
    st.download_button("Download PDF", export_pdf(result), "instrunet_result.pdf")

# ---------------- PAGE ROUTER ----------------
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "login":
    login_page()
elif st.session_state.page == "app":
    if st.session_state.logged_in:
        main_app()
    else:
        st.session_state.page = "login"
        st.rerun()