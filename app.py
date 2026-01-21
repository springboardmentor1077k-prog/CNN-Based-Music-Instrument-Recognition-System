import streamlit as st
import numpy as np
import librosa
import os
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from fpdf import FPDF

# ==========================
# CONFIG
# ==========================
MODEL_PATH = "cnn_music_instruments.keras"
LABEL_MAP_PATH = "label_map.json"

SAMPLE_RATE = 22050
DURATION = 2.5
N_MELS = 64
MAX_FRAMES = 87
TOP_K = 3  # Top K instruments to display
THRESHOLD = 0.3  # Lower threshold = more instruments shown

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

model = load_cnn_model()

# ==========================
# LOAD LABEL MAP
# ==========================
with open(LABEL_MAP_PATH) as f:
    raw_label_map = json.load(f)

# Ensure keys are numeric indices as strings
try:
    label_map = {str(int(k)): v for k, v in raw_label_map.items()}
except ValueError:
    # If keys are instrument names, invert the map
    label_map = {str(i): name for i, name in enumerate(raw_label_map.keys())}

# ==========================
# PREDICTION FUNCTION
# ==========================
def predict_instruments(file_path, top_k=TOP_K):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    if len(audio) < DURATION * sr:
        audio = np.pad(audio, (0, int(DURATION * sr) - len(audio)))

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel)

    if mel_db.shape[1] < MAX_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, MAX_FRAMES - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :MAX_FRAMES]

    mel_db = mel_db[np.newaxis, ..., np.newaxis]
    probs = model.predict(mel_db, verbose=0)[0]

    # Top-K predictions
    top_indices = np.argsort(probs)[::-1][:top_k]
    results = {label_map[str(i)]: float(probs[i]) for i in top_indices if probs[i] >= THRESHOLD}

    return results

# ==========================
# PDF REPORT FUNCTION
# ==========================
def generate_pdf_report(predictions, audio_name):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "CNN-Based Music Instrument Recognition", ln=True, align="C")

    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"Audio File: {audio_name}", ln=True)

    pdf.ln(5)
    pdf.cell(0, 8, "Detected Instruments:", ln=True)

    for inst, conf in predictions.items():
        pdf.cell(80, 8, inst, border=1)
        pdf.cell(40, 8, f"{conf*100:.2f}%", border=1, ln=True)

    # Create a bar chart and save as PNG
    if predictions:
        plt.figure(figsize=(6,4))
        plt.bar(predictions.keys(), [v*100 for v in predictions.values()], color='skyblue')
        plt.ylabel("Confidence %")
        plt.title("Instrument Confidence")
        plt.xticks(rotation=45)
        plt.tight_layout()
        chart_file = "temp_chart.png"
        plt.savefig(chart_file)
        plt.close()
        pdf.image(chart_file, x=15, w=180)
        os.remove(chart_file)

    return pdf.output(dest="S").encode("latin1")

# ==========================
# STREAMLIT UI
# ==========================
st.set_page_config(page_title="Music Instrument Recognition", layout="wide")
st.title("🎵 CNN-Based Music Instrument Recognition System")

uploaded_file = st.file_uploader("Upload WAV audio", type=["wav"])

if uploaded_file:
    temp_file = "temp.wav"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(temp_file)

    predictions = predict_instruments(temp_file)

    if predictions:
        st.subheader("🎯 Detected Instruments")
        for inst, conf in predictions.items():
            st.write(f"**{inst}** : {conf*100:.2f}%")
            st.progress(int(conf*100))

        # JSON download
        st.download_button(
            "📄 Download JSON",
            json.dumps(predictions, indent=4),
            file_name="instrument_report.json",
            mime="application/json"
        )

        # PDF download
        pdf_bytes = generate_pdf_report(predictions, uploaded_file.name)
        st.download_button(
            "📄 Download PDF",
            pdf_bytes,
            file_name="instrument_report.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("No instruments detected above threshold.")

    # Cleanup
    os.remove(temp_file)
