import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="InstruNet AI",
    layout="wide"
)

# =================================================
# AUDIO PREPROCESSING CONFIG (MATCH TRAINING)
# =================================================
TARGET_SR = 16000
TARGET_DURATION = 5.0
MIN_SILENCE_THRESH = 0.01
N_MELS = 128
HOP_LENGTH = 512

# =================================================
# AUDIO PREPROCESSING FUNCTION (FROM TRAINING)
# =================================================
def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None, mono=True)

    # Resample
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    # Peak normalization
    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp

    # Silence trimming
    non_silent = np.where(np.abs(audio) > MIN_SILENCE_THRESH)[0]
    if len(non_silent) > 0:
        audio = audio[non_silent[0]: non_silent[-1]]

    # Fix duration
    target_len = int(TARGET_SR * TARGET_DURATION)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    return audio, sr

# =================================================
# LOGIN PAGE
# =================================================
def login_page():
    st.title("🎵 InstruNet AI")
    st.subheader("CNN-Based Music Instrument Recognition System")
    st.caption("Upload. Analyze. Visualize musical instrument audio.")

    st.markdown("---")

    # ---- CENTERED LOGIN FORM ----
    left, center, right = st.columns([1, 2, 1])

    with center:
        st.markdown("### Login")

        username = st.text_input(
            "Username",
            placeholder="Enter your username"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password"
        )

        st.markdown("")  # spacing

        if st.button("Login", use_container_width=True):
            if username.strip() and password.strip():
                st.session_state.logged_in = True
                st.session_state.user = username
                st.success(f"Welcome, {username}!")
                st.rerun()
            else:
                st.error("Username and password cannot be empty")
# =================================================
# DASHBOARD
# =================================================
def dashboard():

    # -------- SIDEBAR --------
    st.sidebar.title("Dashboard")
    st.sidebar.write(f"Logged in as: **{st.session_state.user}**")
    st.sidebar.write("Upload audio for analysis")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Audio File",
        type=["wav", "mp3"]
    )

    st.sidebar.markdown("---")

    if st.sidebar.button("Sign Out"):
        st.session_state.logged_in = False
        st.session_state.user = None
        st.rerun()

    # -------- MAIN CONTENT --------
    st.title("Audio Analysis Dashboard")
    st.write(
        "Explore waveform and mel-spectrogram representations of uploaded audio samples."
    )

    if uploaded_file is not None:

        # Save uploaded file temporarily
        suffix = uploaded_file.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(uploaded_file.getbuffer())
            audio_path = tmp.name

        # -------- AUDIO PLAYBACK --------
        st.subheader("Audio Playback")
        st.audio(uploaded_file)

        # -------- PREPROCESS AUDIO --------
        y, sr = preprocess_audio(audio_path)

        # -------- WAVEFORM --------
        st.subheader("Waveform Representation")

        fig1, ax1 = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax1)
        ax1.set_title("Audio Waveform")
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Amplitude")
        plt.tight_layout()

        st.pyplot(fig1)
        plt.close(fig1)

        # -------- MEL-SPECTROGRAM --------
        st.subheader("Mel-Spectrogram Representation")

        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=N_MELS,
            hop_length=HOP_LENGTH
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(
            mel_db,
            sr=sr,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="mel",
            ax=ax2
        )
        ax2.set_title("Mel-Spectrogram")
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Mel Frequency")
        fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
        plt.tight_layout()

        st.pyplot(fig2)
        plt.close(fig2)

# =================================================
# APP CONTROLLER
# =================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_page()
else:
    dashboard()
