import streamlit as st
import tempfile
import json
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

from pipeline import run_inference
from Utils.pdf_report import generate_pdf_report
from config import CLASS_NAMES, TARGET_SR
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
st.set_page_config(
    page_title="InstruNet – Instrument Recognition",
    layout="wide"
)

# ==================================================
# TEMP USER STORE (DEMO PURPOSE)
# ==================================================
if "users" not in st.session_state:
    st.session_state.users = {}  # {username: password}

# ==================================================
# SESSION STATE INIT
# ==================================================
for k, v in {
    "authenticated": False,
    "user": None,
    "results": None,
    "pdf_figure": None,
}.items():
    st.session_state.setdefault(k, v)
@st.cache_resource

def load_model():
    return tf.keras.models.load_model(
        "Model/instrunet_irmas_sgd.keras"
    )

# ==================================================
# LOGIN / REGISTER PAGE
# ==================================================
def auth_page():

    st.title("🎵 InstruNet – Music Instrument Recognition")

    mode = st.radio("Choose action", ["Login", "Register"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button(mode):
        if not username.strip() or not password.strip():
            st.error("Username and password cannot be empty.")
            return

        # ---------------- REGISTER ----------------
        if mode == "Register":
            if username in st.session_state.users:
                st.error("User already exists.")
            else:
                st.session_state.users[username] = password
                st.success("Registration successful. Please log in.")

        # ---------------- LOGIN ----------------
        else:
            if username not in st.session_state.users:
                st.error("User does not exist.")
            elif st.session_state.users[username] != password:
                st.error("Incorrect password.")
            else:
                st.session_state.authenticated = True
                st.session_state.user = {"username": username}
                st.success("Login successful!")
                st.rerun()

# ==================================================
# MAIN APP
# ==================================================
def main_app():

    # ---------------- HEADER ----------------
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🎶 InstruNet – Instrument Recognition System")
    with col2:
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()

    # ==================================================
    # SIDEBAR
    # ==================================================
    with st.sidebar:
        st.header("Analysis Settings")

        aggregation = st.radio("Aggregation Method", ["mean", "max"])
        threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.30)
        smoothing = st.slider("Smoothing Window", 1, 7, 1)

        st.divider()
        audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])

        audio_bytes = None
        if audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes)

        run_clicked = st.button("Analyze Track", use_container_width=True)

    # ==================================================
    # AUDIO VISUALIZATION
    # ==================================================
    st.subheader("Audio Visualization")

    if audio_bytes:
        y, sr = librosa.load(BytesIO(audio_bytes), sr=TARGET_SR)

        fig, ax = plt.subplots(figsize=(10, 2))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("Waveform")
        st.pyplot(fig)

        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        fig2, ax2 = plt.subplots(figsize=(10, 3))
        librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax2)
        ax2.set_title("Mel Spectrogram")
        st.pyplot(fig2)

        st.session_state.pdf_figure = fig2
    else:
        st.info("Upload an audio file to begin.")

    # ==================================================
    # INFERENCE
    # ==================================================
    if audio_bytes and run_clicked:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            temp_path = f.name

        model = load_model()

        timeline, times, aggregated, json_out = run_inference(
            temp_path, model, aggregation, threshold, smoothing
        )

        st.session_state.results = {
            "timeline": timeline,      # shape (segments, classes)
            "times": times,            # shape (segments,)
            "aggregated": aggregated,  # shape (classes,)
            "json": json_out
        }

    # ==================================================
    # RESULTS DISPLAY
    # ==================================================
    if st.session_state.results:
        results = st.session_state.results

        # ---------------- PREDICTED INSTRUMENTS ----------------
        st.subheader("🎯 Predicted Instruments")

        preds = results["aggregated"]
        paired = list(zip(CLASS_NAMES, preds))

        cols = st.columns(4)
        shown = 0

        for inst, score in sorted(paired, key=lambda x: -x[1]):
            if score >= threshold:
                cols[shown % 4].metric(
                    label=inst,
                    value=f"{score * 100:.2f}%"
                )
                shown += 1

        if shown == 0:
            st.warning("No instruments passed the confidence threshold.")

        # ==================================================
        # ⏱ INSTRUMENT ACTIVITY TIMELINE
        # ==================================================
        st.markdown("## ⏱ Instrument Activity Timeline")

        timeline = results["timeline"]
        times = results["times"]

        if timeline is None or len(timeline) == 0 or times is None:
            st.info("Temporal timeline unavailable.")
        else:
            fig, ax = plt.subplots(figsize=(12, 4))

            for i, inst in enumerate(CLASS_NAMES):
                activity = timeline[:, i]
                if np.max(activity) >= threshold:
                    ax.plot(
                        times,
                        activity * 100,
                        label=inst,
                        linewidth=2
                    )

            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Prediction (%)")
            ax.set_title("Temporal Instrument Activity")
            ax.set_ylim(0, 100)
            ax.grid(alpha=0.3)
            ax.legend(ncol=3, fontsize=9)

            st.pyplot(fig)

    # ==================================================
    # EXPORT
    # ==================================================
    if st.session_state.results:
        with st.sidebar:
            st.divider()
            st.header("Export Results")

            st.download_button(
                "⬇ Export JSON",
                json.dumps(st.session_state.results["json"], indent=2),
                "analysis.json",
                "application/json",
                use_container_width=True
            )

            if st.session_state.pdf_figure is not None:
                pdf_path = generate_pdf_report(
                    "audio",
                    aggregation,
                    threshold,
                    smoothing,
                    {
                        inst: float(score)
                        for inst, score in zip(CLASS_NAMES, results["aggregated"])
                    },
                    st.session_state.pdf_figure
                )

                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "⬇ Export PDF",
                        f,
                        "analysis.pdf",
                        "application/pdf",
                        use_container_width=True
                    )

# ==================================================
# ENTRY POINT
# ==================================================
if not st.session_state.authenticated:
    auth_page()
else:
    main_app()
