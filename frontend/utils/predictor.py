import numpy as np
import tensorflow as tf
import librosa
import os

class InstrumentPredictor:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = tf.keras.models.load_model(model_path)

        # Class names (MUST match training order)
        self.class_names = [
            "Accordion", "Acoustic_Guitar", "Banjo", "Bass_Guitar", "Clarinet",
            "Cymbals", "Dobro", "Drum_set", "Electro_Guitar", "Floor_Tom",
            "Harmonica", "Harmonium", "Hi_Hats", "Horn", "Keyboard",
            "Mandolin", "Organ", "Piano", "Saxophone", "Shakers",
            "Tambourine", "Trombone", "Trumpet", "Ukulele", "Violin",
            "cowbell", "flute", "vibraphone"
        ]

    # -------------------------------------------------
    def predict_audio_file(
        self,
        audio_path,
        segment_length=3.0,
        smoothing_window=3,
        threshold=0.3
    ):
        y, sr = librosa.load(audio_path, sr=22050, mono=True)

        segment_samples = int(segment_length * sr)
        segments = []

        for i in range(0, len(y) - segment_samples, segment_samples):
            seg = y[i:i + segment_samples]
            mel = librosa.feature.melspectrogram(
                y=seg, sr=sr, n_mels=128
            )
            mel = librosa.power_to_db(mel)
            mel = mel[:128, :128]
            mel = mel[..., np.newaxis]
            segments.append(mel)

        X = np.array(segments)
        preds = self.model.predict(X)

        mean_pred = preds.mean(axis=0)

        detected = []
        for i, name in enumerate(self.class_names):
            conf = mean_pred[i] * 100
            detected.append({
                "name": name,
                "confidence": conf,
                "is_present": conf >= threshold * 100
            })

        timeline = [
            {
                "time": i * segment_length,
                "scores": preds[i].tolist()
            }
            for i in range(len(preds))
        ]

        return {
            "filename": os.path.basename(audio_path),
            "duration": len(y) / sr,
            "sample_rate": sr,
            "detected_instruments": detected,
            "timeline": timeline
        }


# -------------------------------------------------
def create_intensity_bars(detected_instruments):
    bars = {}
    for inst in detected_instruments:
        level = int(inst["confidence"] // 10)
        bars[inst["name"]] = "█" * level
    return bars
