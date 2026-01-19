def build_result(filename, duration, predictions):
    return {
        "metadata": {
            "filename": filename,
            "duration_sec": round(duration, 2)
        },
        "model": {
            "name": "CNN Instrument Classifier"
        },
        "predictions": predictions
    }
