import json
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def export_json(result):
    return json.dumps(result, indent=2)

def export_pdf(result):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    y = 800
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Music Instrument Recognition Report")
    y -= 40

    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"File: {result['metadata']['filename']}")
    y -= 20
    c.drawString(50, y, f"Duration: {result['metadata']['duration_sec']} sec")
    y -= 30

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Detected Instruments:")
    y -= 20

    c.setFont("Helvetica", 10)
    for p in result["predictions"]:
        status = "Detected" if p["detected"] else "Not Detected"
        c.drawString(
            60, y,
            f"{p['instrument']} – {status} (conf={p['confidence']:.2f})"
        )
        y -= 15

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer
