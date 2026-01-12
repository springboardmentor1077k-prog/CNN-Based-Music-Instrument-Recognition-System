import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
MODEL_PATH = "task9/outputs_task9/cnn_model.h5"
# Automatically pick one audio clip (flute example)
FLUTE_DIR = "task16/outputs/segments/flute"

# Automatically pick one audio clip
AUDIO_FILE = os.path.join(
    FLUTE_DIR,
    sorted([f for f in os.listdir(FLUTE_DIR) if f.endswith(".wav")])[0]
)

SR = 22050
SEGMENT_LEN = 3.0
HOP_LEN = 1.5

# Class indices (example – adjust to your model)
INSTRUMENTS = {
    "Flute": 0,
    "Guitar": 1
}


# ---------------- LOAD MODEL ----------------
model = load_model(MODEL_PATH)

# ---------------- LOAD AUDIO ----------------
y, sr = librosa.load(AUDIO_FILE, sr=SR)

segment_samples = int(SEGMENT_LEN * sr)
hop_samples = int(HOP_LEN * sr)

times = []
conf_flute = []
conf_guitar = []

# ---------------- PREDICTIONS OVER TIME ----------------
for i, start in enumerate(range(0, len(y) - segment_samples, hop_samples)):
    segment = y[start:start + segment_samples]

    mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db[:, :130]

    if mel_db.shape[1] < 130:
        mel_db = np.pad(mel_db, ((0, 0), (0, 130 - mel_db.shape[1])))

    mel_db = mel_db.reshape(1, 128, 130, 1)

    pred = model.predict(mel_db, verbose=0)[0]

    times.append(i * HOP_LEN)
    conf_flute.append(pred[INSTRUMENTS["Flute"]])
    conf_guitar.append(pred[INSTRUMENTS["Guitar"]])

# ---------------- PLOT ----------------
plt.figure(figsize=(10, 5))

plt.plot(times, conf_flute, label="Flute", linewidth=2)
plt.plot(times, conf_guitar, label="Guitar", linewidth=2)

plt.axhline(0.5, linestyle="--", linewidth=1)

plt.xlabel("Time (seconds)")
plt.ylabel("Confidence")
plt.title("Instrument Timeline (Model Predictions)")
plt.legend()
plt.tight_layout()

os.makedirs("task18/outputs", exist_ok=True)
plt.savefig("task18/outputs/instrument_timeline.png")
plt.show()
