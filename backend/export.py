import numpy as np
import json
from fpdf import FPDF
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tempfile
from backend.utils import load_labels

def safe_text(text: str) -> str:
    return text.encode("latin-1", "ignore").decode("latin-1")


def export_json(result):
    """Export the pipeline result as JSON"""
    return json.dumps(result, indent=4)

def export_pdf(result):
    """Export the pipeline result as PDF with optional mel-spectrogram and segment info"""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "InstruNet AI - Music Instrument Recognition", ln=True)
    
    # File details
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, safe_text(f"File: {result['metadata']['file_name']}"), ln=True)

    pdf.cell(0, 8, f"Duration: {result['metadata']['duration_sec']:.2f}s", ln=True)
    pdf.cell(0, 8, f"Sample Rate: {result['metadata']['sample_rate']}", ln=True)
    
    # Detected instruments
    pdf.cell(
    0,
    8,
    safe_text("Detected Instruments: " + ", ".join(result['predictions']['final_instruments'])
    ),
    ln=True
)

    
    # Confidence scores
    pdf.cell(0, 8, "Confidence Scores:", ln=True)
    for inst, conf in result['predictions']['confidence_scores'].items():
        pdf.cell(
    0,
    6,
    safe_text(f"{inst}: {conf:.2f}"),
    ln=True
)
    
    # Segment timelines
    pdf.cell(0, 8, "Segment-wise Probabilities:", ln=True)
    labels = load_labels()
    for i, seg_probs in enumerate(result['predictions']['timelines']):
        seg_text = ", ".join([f"{lbl}: {prob:.2f}" for lbl, prob in zip(labels, seg_probs)])
        pdf.multi_cell(0, 6, safe_text(f"Segment {i+1}: {seg_text}"))

    
    # Inference steps
    pdf.cell(0, 8, "Inference Steps:", ln=True)
    steps = result.get(
    "inference_info", {}
).get(
    "steps",
    "Audio loaded -> Mel spectrogram -> CNN inference -> Postprocessing -> Result returned"
)
    pdf.multi_cell(0, 6, safe_text(steps))
    
    # Optional: mel-spectrogram
    try:
        y, sr = librosa.load(result['metadata']['file_name'], sr=None)
        S = librosa.feature.melspectrogram(y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots(figsize=(6,3))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='magma', ax=ax)
        tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmp_img.name, bbox_inches='tight')
        plt.close(fig)
        pdf.image(tmp_img.name, x=10, y=pdf.get_y(), w=pdf.w-20)
    except:
        pass

    return pdf.output(dest='S').encode('latin1')
