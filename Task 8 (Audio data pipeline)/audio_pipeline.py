# =========================================
# End-to-End Audio Data Pipeline (IRMAS)
# =========================================

import os
import librosa
import numpy as np

# ---------- Configuration ----------
DATASET_ROOT = "IRMAS-TrainingData"
OUTPUT_DIR = "cnn_features"

SAMPLE_RATE = 22050
DURATION = 3.0
SAMPLES = int(SAMPLE_RATE * DURATION)

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Audio Loading ----------
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return y, sr

# ---------- Silence Trimming ----------
def trim_silence(y):
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    return y_trimmed

# ---------- Duration Fixing ----------
def fix_duration(y):
    if len(y) < SAMPLES:
        y = np.pad(y, (0, SAMPLES - len(y)))
    else:
        y = y[:SAMPLES]
    return y

# ---------- Normalization ----------
def normalize_audio(y):
    return librosa.util.normalize(y)

# ---------- Data Augmentation ----------
def pitch_shift(y, sr, steps=2):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

def time_stretch(y, rate=1.2):
    return librosa.effects.time_stretch(y, rate=rate)

def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

# ---------- Mel Spectrogram ----------
def extract_mel_spectrogram(y, sr):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

# ---------- Complete Pipeline ----------
def audio_pipeline(file_path):
    y, sr = load_audio(file_path)
    y = trim_silence(y)
    y = fix_duration(y)
    y = normalize_audio(y)

    features = []

    # Original
    features.append(extract_mel_spectrogram(y, sr))

    # Augmented versions
    features.append(extract_mel_spectrogram(pitch_shift(y, sr), sr))
    features.append(extract_mel_spectrogram(time_stretch(y), sr))
    features.append(extract_mel_spectrogram(add_noise(y), sr))

    return features

# ---------- Process IRMAS Dataset ----------
X = []
y_labels = []
label_map = {}
label_id = 0

print("Starting audio data pipeline...")

for root, _, files in os.walk(DATASET_ROOT):
    for file in files:
        if not file.lower().endswith(".wav"):
            continue

        file_path = os.path.join(root, file)

        # Extract IRMAS label from filename [pia], [vio], etc.
        label = file[file.find("[")+1:file.find("]")]

        if label not in label_map:
            label_map[label] = label_id
            label_id += 1

        mel_features = audio_pipeline(file_path)

        for mel in mel_features:
            # Ensure fixed shape
            if mel.shape[1] < 128:
                continue
            mel = mel[:, :128]

            X.append(mel)
            y_labels.append(label_map[label])

print("Pipeline completed")

# ---------- Convert to CNN-ready format ----------
X = np.array(X)
y_labels = np.array(y_labels)

# Add channel dimension
X = X[..., np.newaxis]

# ---------- Save Features ----------
np.save(os.path.join(OUTPUT_DIR, "X_mel.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y_labels.npy"), y_labels)

print("Saved CNN-ready features")
print("X shape:", X.shape)
print("y shape:", y_labels.shape)
print("Label map:", label_map)
