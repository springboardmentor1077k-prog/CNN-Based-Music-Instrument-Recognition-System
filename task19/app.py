import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Music Instrument Recognition",
    layout="wide"
)

# ---------------- Title & Description ----------------
st.title("Music Instrument Recognition")
st.caption("Visual exploration of audio signals for instrument recognition.")

# ---------------- Sidebar ----------------
st.sidebar.header("Audio Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3"]
)

if uploaded_file:
    st.sidebar.audio(uploaded_file)

    # ---------------- Load Audio ----------------
    y, sr = librosa.load(uploaded_file, sr=None)
    duration = len(y) / sr
    time = np.linspace(0, duration, len(y))

    st.divider()

    # ---------------- Waveform ----------------
    st.subheader("Waveform")

    fig1, ax1 = plt.subplots(figsize=(12, 3))
    ax1.plot(time, y, color="#1f77b4")
    ax1.set_title("Audio Waveform")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(alpha=0.3)
    st.pyplot(fig1, use_container_width=True)

    st.divider()

    # ---------------- Mel Spectrogram ----------------
    st.subheader("Mel Spectrogram")

    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        ax=ax2
    )
    ax2.set_title("Mel Spectrogram (dB)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Mel Frequency")
    fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
    st.pyplot(fig2, use_container_width=True)
