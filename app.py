# app.py

import streamlit as st
import tempfile
import json
import tensorflow as tf

from pipeline import run_inference
from utils.visualization import plot_intensity
from utils.pdf_report import generate_pdf_report
from config import CLASS_NAMES

# --------------------------------------------------
# Session State Initialization
# --------------------------------------------------

if "results" not in st.session_state:
    st.session_state.results = None

# --------------------------------------------------
# Page Configuration & Global Styling
# --------------------------------------------------

st.set_page_config(
    page_title="InstruNet AI – Music Instrument Recognition",
    layout="wide"
)

st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.block-container {
    padding: 2rem;
}
h1, h2, h3 {
    color: #ffffff;
}
.small-label {
    color: #9aa0a6;
    font-size: 0.85rem;
}
.card {
    background-color: #161b22;
    padding: 1.2rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Model Loader (Cached)
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/best_l2_regularized_model.h5"
    )

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("InstruNet AI: Music Instrument Recognition")
st.markdown(
    "Upload an audio file to analyze instrument presence, confidence, and temporal intensity."
)

# --------------------------------------------------
# Layout: Left | Center | Right
# --------------------------------------------------

left, center, right = st.columns([1.2, 3, 1.6])

# ==================================================
# LEFT PANEL — Upload & Controls
# ==================================================

with left:
    st.markdown("### 🎵 Upload Audio")

    audio_file = st.file_uploader(
        "Choose WAV or MP3 file",
        type=["wav", "mp3"]
    )

    if audio_file:
        st.audio(audio_file)
        st.markdown(
            f"<div class='small-label'>Now Playing:</div><b>{audio_file.name}</b>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### ⚙️ Analysis Settings")

    aggregation = st.selectbox(
        "Aggregation Method",
        ["mean", "max", "voting"]
    )

    threshold = st.slider(
        "Confidence Threshold",
        0.0, 1.0, 0.25
    )

    smoothing = st.slider(
        "Smoothing Window",
        1, 7, 3
    )

    run_clicked = st.button("▶ Analyze Track", use_container_width=True)

# ==================================================
# CENTER PANEL — Placeholder
# ==================================================

with center:
    st.markdown("### 📊 Analysis Results")
    if not audio_file and st.session_state.results is None:
        st.info("Upload an audio file and click Analyze Track.")

# ==================================================
# RIGHT PANEL — Placeholder
# ==================================================

with right:
    st.markdown("### 🎼 Detected Instruments")

# --------------------------------------------------
# Run Inference (ONCE) and Store in Session State
# --------------------------------------------------

if audio_file and run_clicked:
    with st.spinner("Running inference pipeline..."):

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(audio_file.read())
            temp_path = f.name

        model = load_model()

        intensities, times, aggregated, json_out = run_inference(
            temp_path,
            model,
            aggregation,
            threshold,
            smoothing
        )

        st.session_state.results = {
            "intensities": intensities,
            "times": times,
            "aggregated": aggregated,
            "json": json_out
        }

# --------------------------------------------------
# Render Results (FROM SESSION STATE)
# --------------------------------------------------

if st.session_state.results is not None:

    results = st.session_state.results
    intensities = results["intensities"]
    times = results["times"]
    aggregated = results["aggregated"]
    json_out = results["json"]

    # --------------------------------------------------
    # CENTER: Timeline Plot
    # --------------------------------------------------

    with center:
        st.markdown("#### Instrument Intensity Timeline")

        selected_instruments = st.multiselect(
            "Select instruments to display",
            CLASS_NAMES,
            default=CLASS_NAMES
        )

        if selected_instruments:
            fig = plot_intensity(
                times,
                intensities[:, [CLASS_NAMES.index(i) for i in selected_instruments]],
                threshold
            )
            st.pyplot(fig)
        else:
            st.warning("Select at least one instrument to visualize.")

    # --------------------------------------------------
    # RIGHT: Confidence Summary & Export
    # --------------------------------------------------

    with right:
        st.markdown("#### Confidence Summary")

        confidence_dict = {
            cls: float(aggregated[i])
            for i, cls in enumerate(CLASS_NAMES)
            if aggregated[i] >= threshold
        }

        if confidence_dict:
            for cls, score in sorted(
                confidence_dict.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                st.markdown(f"**{cls.upper()}**")
                st.progress(min(score, 1.0))
        else:
            st.warning("No instruments exceeded the threshold.")

        st.markdown("---")
        st.markdown("### 📤 Export Results")

        st.download_button(
            "Export JSON",
            data=json.dumps(json_out, indent=2),
            file_name="analysis.json",
            mime="application/json",
            use_container_width=True
        )

        pdf_path = generate_pdf_report(
            audio_name=audio_file.name if audio_file else "audio",
            aggregation=aggregation,
            threshold=threshold,
            smoothing=smoothing,
            confidence_dict=confidence_dict,
            plot_figure=fig
        )

        with open(pdf_path, "rb") as f:
            st.download_button(
                "Export PDF",
                data=f,
                file_name="analysis.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    st.success("Analysis completed successfully.")
