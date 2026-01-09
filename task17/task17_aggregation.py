import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
MODEL_PATH = "task9/outputs_task9/cnn_model.h5"
SEGMENTS_DIR = "task16/outputs/segments"
OUTPUT_DIR = "task17/outputs"
SR = 22050

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
model = load_model(MODEL_PATH)

# Class order (must match training)
CLASSES = ["flute", "guitar", "piano"]

segment_preds = {}   # clip_id -> list of softmax vectors

# ------------- SEGMENT-LEVEL PREDICTION -------------
for instrument in os.listdir(SEGMENTS_DIR):
    inst_path = os.path.join(SEGMENTS_DIR, instrument)

    for file in os.listdir(inst_path):
        if not file.lower().endswith(".wav"):
            continue

        file_path = os.path.join(inst_path, file)

        # Clip ID (group segments of same clip)
        clip_id = "_".join(file.split("_")[:-1])

        # Load audio
        y, sr = librosa.load(file_path, sr=SR)

        # Mel spectrogram (same as training)
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Ensure fixed size (128, 130)
        mel_db = mel_db[:, :130]

        # Pad if needed
        if mel_db.shape[1] < 130:
            pad_width = 130 - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")

        # Reshape for CNN
        mel_db = mel_db.reshape(1, 128, 130, 1)

        pred = model.predict(mel_db, verbose=0)[0]

        segment_preds.setdefault(clip_id, []).append(pred)

# ------------- NO AGGREGATION ----------------
with open(os.path.join(OUTPUT_DIR, "clip_predictions_no_agg.txt"), "w") as f:
    for clip_id, preds in segment_preds.items():
        label_idx = int(np.argmax(preds[0]))
        f.write(f"{clip_id} -> class_{label_idx}\n")


# ------------- WITH AGGREGATION (AVERAGING) -------------
with open(os.path.join(OUTPUT_DIR, "clip_predictions_with_agg.txt"), "w") as f:
    for clip_id, preds in segment_preds.items():
        avg_pred = np.mean(preds, axis=0)
        label_idx = int(np.argmax(avg_pred))
        confidence = float(np.max(avg_pred))
        f.write(f"{clip_id} -> class_{label_idx} (conf={confidence:.2f})\n")

print("✅ Task 17 aggregation completed")
