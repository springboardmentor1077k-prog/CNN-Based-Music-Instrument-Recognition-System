import json
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def export_json(result):
    return json.dumps(result, indent=2)

def export_pdf(result):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = [
        Paragraph("InstruNet AI – Instrument Recognition Report", styles["Title"]),
        Paragraph(f"Audio: {result['metadata']['file_name']}", styles["Normal"]),
        Paragraph(f"Detected Instruments: {', '.join(result['predictions']['final_instruments'])}", styles["Normal"]),
    ]

    doc.build(content)
    return buffer.getvalue()
