from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
import tempfile
import os

from inference import run_inference, CLASS_NAMES
from aggregation import aggregate_predictions

app = FastAPI(title="InstruNet AI Backend")


@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    segment_length: float = Form(3.0),
    aggregation: str = Form("mean"),
    top_k: int = Form(5)
):
    # -----------------------------
    # 1️⃣ Save uploaded audio
    # -----------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        audio_path = tmp.name

    try:
        # -----------------------------
        # 2️⃣ Run model inference
        # -----------------------------
        # preds shape: (num_segments, num_classes)
        preds = run_inference(audio_path, segment_length)

        if preds is None or len(preds) == 0:
            return {"predictions": []}

        # -----------------------------
        # 3️⃣ Aggregate predictions
        # -----------------------------
        aggregated = aggregate_predictions(preds, method=aggregation)

        # Ensure numpy array
        aggregated = np.asarray(aggregated).flatten()

        # -----------------------------
        # 4️⃣ Top-K selection
        # -----------------------------
        top_indices = np.argsort(aggregated)[::-1][:top_k]

        predictions = []
        for idx in top_indices:
            predictions.append({
                "instrument": CLASS_NAMES[idx],
                "confidence": float(aggregated[idx])
            })

        # -----------------------------
        # 5️⃣ Final response (🔥 THIS FIXES EVERYTHING)
        # -----------------------------
        return {
            "predictions": predictions
        }

    finally:
        os.remove(audio_path)
