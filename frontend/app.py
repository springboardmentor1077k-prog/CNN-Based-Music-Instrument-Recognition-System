import sys
from pathlib import Path
import numpy as np   # ✅ FIXED: numpy import added
import streamlit as st
import json
import tempfile
import matplotlib.pyplot as plt
import librosa
import librosa.display

# -------------------------------------------------
# backend import path
# -------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from backend.api import run_inference

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="InstruNet AI",
    layout="wide"
)

# =================================================
# LOGIN PAGE
# =================================================
def login_page():
    st.title("🎵 InstruNet AI")
    st.subheader("CNN-Based Music Instrument Recognition System")
    st.caption("Upload audio • Analyze • Detect instruments")

    st.markdown("---")

    _, center, _ = st.columns([1, 2, 1])
    with center:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            if username.strip() and password.strip():
                st.session_state.logged_in = True
                st.session_state.user = username
                st.rerun()
            else:
                st.error("Username and password cannot be empty")

# =================================================
# DASHBOARD
# =================================================
def dashboard():

    # ---------- SIDEBAR ----------
    st.sidebar.title("Dashboard")
    st.sidebar.write(f"👤 {st.session_state.user}")

    uploaded_file = st.sidebar.file_uploader(
        "Upload audio file",
        type=["wav", "mp3"]
    )

    if st.sidebar.button("Sign Out"):
        st.session_state.logged_in = False
        st.session_state.user = None
        st.rerun()

    # ---------- MAIN ----------
    st.title("🎧 Instrument Recognition Results")

    if not uploaded_file:
        st.info("Upload an audio file to begin analysis.")
        return

    # ---------- SAVE FILE ----------
    suffix = uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(uploaded_file.getbuffer())
        audio_path = tmp.name

    # ---------- AUDIO ----------
    st.subheader("Audio Preview")
    st.audio(uploaded_file)

    # ---------- MEL-SPECTROGRAM ----------
    st.subheader("Mel-Spectrogram")

    y, sr = librosa.load(audio_path, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)  # ✅ FIXED

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax
    )
    ax.set_title("Mel-Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)
    plt.close(fig)

    # ---------- INFERENCE ----------
    with st.spinner("Analyzing audio..."):
        result = run_inference(audio_path)

    aggregated = result["aggregated"]
    segments = result["segments"]

    # ---------- CONFIDENCE BAR CHART ----------
    st.subheader("Analysis Results")

    labels = list(aggregated.keys())
    values = list(aggregated.values())

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.barh(labels, values)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Confidence")
    ax2.set_title("Instrument Confidence")
    st.pyplot(fig2)
    plt.close(fig2)

    # ---------- DETECTED INSTRUMENTS ----------
    st.subheader("Detected Instruments")

    cols = st.columns(3)
    for i, (inst, prob) in enumerate(aggregated.items()):
        cols[i % 3].checkbox(
            f"{inst.upper()} ({prob:.2f})",
            value=prob >= 0.3,
            disabled=True
        )

    # ---------- INSTRUMENT TIMELINE ----------
    st.subheader("Instrument Timeline")

    times = [s["start_time_sec"] for s in segments]

    fig3, ax3 = plt.subplots(figsize=(10, 3))
    for inst in aggregated.keys():
        probs = [s["probabilities"][inst] for s in segments]
        ax3.plot(times, probs, label=inst)

    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Probability")
    ax3.legend(ncol=3, fontsize=8)
    st.pyplot(fig3)
    plt.close(fig3)

    # ---------- EXPORT ----------
    st.subheader("Export Report")

    st.download_button(
        "⬇ Download JSON Report",
        data=json.dumps(result, indent=4),
        file_name="instrument_results.json",
        mime="application/json"
    )

# =================================================
# APP CONTROLLER
# =================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_page()
else:
    dashboard()
