from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import tempfile
import os

def export_pdf(results: dict) -> str:
    """
    Generates a clean PDF report for InstruNet AI
    Returns path to generated PDF
    """

    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4

    y = height - 50

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, "InstruNet AI — Instrument Recognition Report")
    y -= 40

    # Detected instruments
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Detected Instruments")
    y -= 20

    c.setFont("Helvetica", 12)

    for inst in results["top_predictions"]:
        line = f"{inst['name']} : {inst['confidence']*100:.2f}%"
        c.drawString(60, y, line)
        y -= 18

        if y < 100:
            c.showPage()
            y = height - 50

    # Metadata
    y -= 30
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Analysis Settings")
    y -= 20

    c.setFont("Helvetica", 12)
    c.drawString(60, y, f"Aggregation Method: {results['aggregation']}")
    y -= 18
    c.drawString(60, y, f"Confidence Threshold: {results['threshold']}")

    c.save()
    return path
