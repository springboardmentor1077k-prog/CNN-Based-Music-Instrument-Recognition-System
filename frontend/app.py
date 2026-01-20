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
.stAlert,
.stSuccess,
.stError,
.stWarning,
.stInfo {
    background: rgba(59,130,246,0.18) !important;
    color: white !important;
    border-radius: 14px !important;
}

.stAlert * {
    color: white !important;
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
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Username and password cannot be empty")

# ---------------- DASHBOARD ----------------
def main_app():
    with st.sidebar:
        # User info
        st.markdown(f"<div style='text-align:center; margin-bottom:20px;'>"
                    f"<div style='width:70px;height:70px;background-color:#1E90FF;border-radius:50%;display:flex;justify-content:center;align-items:center;font-size:28px;color:white;margin:0 auto;'>{st.session_state.user[0].upper()}</div>"
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
            st.session_state.logged_in = False
            st.session_state.page = "home"
            st.rerun()

    # Main content
    st.title("🎵 InstruNet AI")
    if audio is None:
        st.info("Upload an audio file to begin.")
        st.stop()

    st.audio(audio)

    if run:
        with st.spinner("Running inference..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio.read())
                path = tmp.name
            st.session_state.result = run_pipeline(
                audio_path=path,
                original_filename=audio.name,
                threshold=threshold,
                aggregation=aggregation
            )

    result = st.session_state.get("result")
    if result is None:
        st.info("Click Predict to analyze.")
        st.stop()

    st.markdown("### 🎯 Detected Instruments")
    final = result["predictions"]["final_instruments"]
    if not final:
        st.warning("No dominant instruments detected.")
    else:
        for inst in final:
            st.success(inst.upper())

    with st.expander("Confidence scores"):
        st.bar_chart(result["predictions"]["confidence_scores"])

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
