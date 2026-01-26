import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import tempfile
import matplotlib.pyplot as plt
import os
import json

try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


# ==============================
# PAGE CONFIG (ONLY ONCE)
# ==============================
st.set_page_config(
    page_title="InstruNet-AI",
    page_icon="🎵",
    layout="wide"
)

# ==============================
# USER CREDENTIALS (DEMO)
# ==============================
USERS = {
    "admin": "admin123",
    "srishanth": "instru123"
}

# ==============================
# SESSION STATE INIT
# ==============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = ""


# ==============================
# LOGIN UI
# ==============================
def login_ui():
    st.markdown(
        """
        <style>
        .login-box {
            max-width: 420px;
            margin: auto;
            padding: 30px;
            background: #111827;
            border-radius: 14px;
            border: 1px solid #1f2937;
            box-shadow: 0px 8px 20px rgba(0,0,0,0.4);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='login-box'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center'>🔐 InstruNet Login</h2>", unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Username").strip()
        password = st.text_input("Password", type="password").strip()
        login_btn = st.form_submit_button("Login")

    if login_btn:
        if username in USERS and USERS[username] == password:
            st.session_state.logged_in = True
            st.session_state.user = username
            st.success("✅ Login successful")
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

    st.markdown("</div>", unsafe_allow_html=True)


# ==============================
# AUTH GATE
# ==============================
if not st.session_state.logged_in:
    login_ui()
    st.stop()


# ==============================
# MAIN APPLICATION STARTS HERE
# ==============================

# ------------------------------
# UI STYLING
# ------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0f172a; color: #e5e7eb; }
    html, body, [class*="css"] { color: #e5e7eb !important; }
    h1, h2, h3, h4 { color: #f9fafb !important; font-weight: 700; }
    .subtitle { color: #9ca3af; font-size: 16px; }
    .card {
        background-color: #111827;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #1f2937;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.4);
        margin-bottom: 20px;
    }
    section[data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 1px solid #1f2937;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# HEADER
# ------------------------------
st.markdown("<h1>🎵 InstruNet AI: Music Instrument Recognition</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload. Analyze. Discover.</div>", unsafe_allow_html=True)

# ------------------------------
# CONSTANTS
# ------------------------------
SAMPLE_RATE = 16050
DURATION = 3
SAMPLES = SAMPLE_RATE * DURATION
N_MELS = 128
FIXED_FRAMES = 128

CLASSES = ["cel","cla","flu","gac","gel","org","pia","sax","tru","vio","voi"]

LABEL_MAP = {
    "cel": "Cello", "cla": "Clarinet", "flu": "Flute",
    "gac": "Acoustic Guitar", "gel": "Electric Guitar",
    "org": "Organ", "pia": "Piano", "sax": "Saxophone",
    "tru": "Trumpet", "vio": "Violin", "voi": "Voice"
}

# ------------------------------
# SIDEBAR
# ------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.caption(f"👤 Logged in as: {st.session_state.user}")

    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.user = ""
        st.rerun()

    top_k = st.slider("Top-K Instruments", 1, 6, 4)
    st.markdown("---")
    st.caption("Softmax Top-K ranking")

# ------------------------------
# LOAD MODEL
# ------------------------------
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "irmas_cnn_final.keras")
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ------------------------------
# PREPROCESS
# ------------------------------
def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    y = np.pad(y, (0, max(0, SAMPLES - len(y))))[:SAMPLES]

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_db = np.pad(mel_db, ((0,0),(0, max(0, FIXED_FRAMES - mel_db.shape[1]))))[:, :FIXED_FRAMES]

    mel_norm = (mel_db - np.mean(mel_db)) / np.std(mel_db)
    mel_norm = mel_norm[..., np.newaxis]
    mel_norm = np.expand_dims(mel_norm, axis=0)

    return mel_norm, mel_db

# ------------------------------
# PREDICTION
# ------------------------------
def predict_top_k(file_path, k):
    X, spec_db = preprocess_audio(file_path)
    probs = model.predict(X, verbose=0)[0]
    indices = probs.argsort()[-k:][::-1]
    return [(CLASSES[i], probs[i]) for i in indices], probs, spec_db

# ------------------------------
# LAYOUT
# ------------------------------
col_left, col_mid, col_right = st.columns([1.2, 2.5, 1.3])

# ------------------------------
# UPLOAD
# ------------------------------
with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⬆️ Upload Audio")
    uploaded_file = st.file_uploader("Choose File", type=["wav","mp3"])
    if uploaded_file:
        st.audio(uploaded_file)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------
# ANALYSIS
# ------------------------------
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    top_results, probs, spec_db = predict_top_k(audio_path, top_k)
    os.remove(audio_path)

    with col_mid:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Analysis Results")

        fig, ax = plt.subplots(figsize=(9,4))
        img = librosa.display.specshow(
            spec_db, sr=SAMPLE_RATE, x_axis="time", y_axis="mel", ax=ax
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        st.pyplot(fig)

        st.subheader("📊 Instrument Confidence Scores")
        for name, score in top_results:
            pct = score * 100
            st.markdown(
                f"<b>{LABEL_MAP[name]}</b> — "
                f"<span style='color:#38bdf8'>{pct:.2f}%</span>",
                unsafe_allow_html=True
            )
            st.progress(int(pct))

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        dom, dom_score = top_results[0]
        st.success(f"🎼 Dominant Instrument: **{LABEL_MAP[dom]}** ({dom_score*100:.2f}%)")

        st.markdown("### Present Instruments")
        for name, _ in top_results:
            st.write(f"✔ {LABEL_MAP[name]}")

        st.download_button(
            "⬇️ Download JSON",
            json.dumps(
                {
                    "top_k": [
                        {"instrument": LABEL_MAP[n], "confidence": float(c)}
                        for n, c in top_results
                    ]
                },
                indent=4
            ),
            file_name="prediction.json",
            mime="application/json"
        )

        if not PDF_AVAILABLE:
            st.info("📄 PDF export unavailable (fpdf not installed)")
