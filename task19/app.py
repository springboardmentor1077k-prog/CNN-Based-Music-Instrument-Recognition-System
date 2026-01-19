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

from pipeline import predict_stub
from utils.results import build_result
from utils.export import export_json, export_pdf
import pandas as pd

# ---- Predict Button ----
st.divider()
predict_clicked = st.button("Predict")

if predict_clicked:
    predictions = predict_stub()

    result = build_result(
        filename=uploaded_file.name,
        duration=duration,
        predictions=predictions
    )

    # ---- Prediction Results ----
    st.header("Prediction Results")

    # Primary result
    for p in predictions:
        status = "Detected" if p["detected"] else "Not Detected"
        st.markdown(f"**{p['instrument']} – {status}**")

    st.subheader("Confidence Levels")
    for p in predictions:
        st.progress(p["confidence"])

    # Structured table
    st.subheader("Detailed View")
    df = pd.DataFrame(predictions)
    st.table(df)

    # Advanced view
    with st.expander("View all probabilities"):
        st.json(result["predictions"])

    # ---- Exports ----
    st.download_button(
        "Download JSON",
        export_json(result),
        file_name="prediction_result.json",
        mime="application/json"
    )

    pdf_buffer = export_pdf(result)
    st.download_button(
        "Download PDF Report",
        pdf_buffer,
        file_name="prediction_report.pdf",
        mime="application/pdf"
    )
