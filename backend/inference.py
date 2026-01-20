import os
import numpy as np
import librosa
import tensorflow as tf

# -------------------------------------------------
# Paths (SAFE & EXPLICIT)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "..", "cnn_new", "model_instrunet_phase2.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR, "..", "cnn_new", "features", "class_names.npy"
)

# -------------------------------------------------
# Load model
# -------------------------------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded")

# -------------------------------------------------
# Load class names
# -------------------------------------------------
if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"class_names.npy not found at {CLASS_NAMES_PATH}"
    )

CLASS_NAMES = np.load(CLASS_NAMES_PATH, allow_pickle=True)
CLASS_NAMES = CLASS_NAMES.tolist()

print(f"✅ Loaded {len(CLASS_NAMES)} classes")


# -------------------------------------------------
# Feature extraction (MEL)
# -------------------------------------------------
def extract_mel(y, sr, n_mels=128, max_len=128):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < max_len:
        pad = max_len - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)))
    else:
        mel_db = mel_db[:, :max_len]

    return mel_db


# -------------------------------------------------
# 🔮 Run inference
# -------------------------------------------------
def run_inference(audio_path, segment_length=3.0):
    """
    Returns:
        np.ndarray (num_segments, num_classes)
    """
    y, sr = librosa.load(audio_path, sr=None)

    segment_samples = int(segment_length * sr)
    preds = []

    for start in range(0, len(y), segment_samples):
        segment = y[start:start + segment_samples]

        if len(segment) < segment_samples:
            continue

        mel = extract_mel(segment, sr)
        mel = mel[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)

        prob = model.predict(mel, verbose=0)[0]
        preds.append(prob)

    return np.array(preds)
