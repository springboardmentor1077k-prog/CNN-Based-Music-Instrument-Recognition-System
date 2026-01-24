from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import tempfile
import os

def generate_pdf_report(
    audio_name,
    aggregation,
    threshold,
    smoothing,
    confidence_dict,
    plot_figure
):
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, f"{audio_name}_analysis.pdf")
    plot_path = os.path.join(temp_dir, "intensity_plot.png")

    # Save matplotlib figure
    plot_figure.savefig(plot_path, dpi=200, bbox_inches="tight")

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "InstruNet – Instrument Recognition Report")

    # Metadata
    c.setFont("Helvetica", 10)
    y = height - 90
    c.drawString(50, y, f"Audio File: {audio_name}")
    y -= 15
    c.drawString(50, y, f"Aggregation Method: {aggregation}")
    y -= 15
    c.drawString(50, y, f"Threshold: {threshold}")
    y -= 15
    c.drawString(50, y, f"Smoothing Window: {smoothing}")

    # Confidence summary
    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Detected Instruments (Above Threshold):")

    y -= 20
    c.setFont("Helvetica", 10)

    # ===============================
    # Handle confidence input safely
    # ===============================

    confidence_pairs = []

    if confidence_dict is None:
        confidence_pairs = []

    elif isinstance(confidence_dict, dict):
        confidence_pairs = confidence_dict.items()

    else:
        # Assume numpy array aligned with CLASS_NAMES
        from config import CLASS_NAMES
        confidence_pairs = zip(CLASS_NAMES, confidence_dict)

    # ===============================
    # Render results
    # ===============================
    if confidence_pairs:
        for inst, score in confidence_pairs:
            score = float(score)
            if score > 0:
                c.drawString(60, y, f"{inst.upper()} : {score * 100:.2f}%")
                y -= 15
    else:
        c.drawString(60, y, "No instruments exceeded the threshold.")

    # Plot
    y -= 20
    c.drawImage(
        plot_path,
        50,
        y - 250,
        width=width - 100,
        height=250,
        preserveAspectRatio=True
    )

    c.showPage()
    c.save()

    return pdf_path