import sys
from pathlib import Path

import numpy as np
import librosa
import tensorflow as tf

# ---------------- Constants ----------------
SR = 16000
SEGMENT_DURATION = 2.0
HOP_DURATION = 1.0

SEGMENT_SAMPLES = int(SEGMENT_DURATION * SR)
HOP_SAMPLES = int(HOP_DURATION * SR)

N_MELS = 128
HOP_LENGTH = 512
IMG_SIZE = 128

INSTRUMENTS = [
    "cel", "cla", "flu", "gac", "gel",
    "org", "pia", "sax", "tru", "vio", "voi"
]

# ---------------- Load Model ----------------
# Path to your saved Keras model (.keras file)
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "multilabel_cnn_improved.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------- Audio Utils ----------------
def load_audio(file):
    audio, _ = librosa.load(file, sr=SR, mono=True)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    return audio


def segment_audio(audio):
    segments, times = [], []
    for start in range(0, len(audio) - SEGMENT_SAMPLES + 1, HOP_SAMPLES):
        segments.append(audio[start:start + SEGMENT_SAMPLES])
        times.append(start / SR)
    return segments, times


def segment_to_mel(segment):
    mel = librosa.feature.melspectrogram(
        y=segment, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_db = tf.image.resize(
        mel_db[..., None], (IMG_SIZE, IMG_SIZE)
    ).numpy()

    mel_db = np.repeat(mel_db, 3, axis=-1)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

    return mel_db

# ---------------- Prediction ----------------
def predict(audio):
    segments, times = segment_audio(audio)
    segment_probs = []

    for seg in segments:
        mel = segment_to_mel(seg)
        mel = mel[None, ...]  # Add batch dimension
        segment_probs.append(model.predict(mel, verbose=0)[0])

    segment_probs = np.array(segment_probs)

    # Top-K aggregation per instrument
    agg_probs = []
    for c in range(segment_probs.shape[1]):
        topk = np.sort(segment_probs[:, c])[-3:]  # top 3 values
        agg_probs.append(float(np.mean(topk)))

    return {
        "segments": build_segments_json(segment_probs, times),
        "aggregated": dict(zip(INSTRUMENTS, agg_probs))
    }


def build_segments_json(segment_probs, times):
    segments = []
    for i, (t, probs) in enumerate(zip(times, segment_probs)):
        segments.append({
            "segment_index": i,
            "start_time_sec": round(t, 2),
            "end_time_sec": round(t + SEGMENT_DURATION, 2),
            "probabilities": {
                INSTRUMENTS[j]: float(probs[j])
                for j in range(len(INSTRUMENTS))
            }
        })
    return segments