import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# =========================
# PATH CONFIG (FIXED)
# =========================
BASE_DIR = r"C:\Users\ADMIN\Downloads\check\cnn_new"

CSV_DIR = os.path.join(BASE_DIR, "data", "audio_csv")
FEATURE_DIR = os.path.join(BASE_DIR, "features")

TRAIN_CSV = os.path.join(CSV_DIR, "train.csv")
VAL_CSV   = os.path.join(CSV_DIR, "val.csv")
TEST_CSV  = os.path.join(CSV_DIR, "test.csv")

os.makedirs(FEATURE_DIR, exist_ok=True)

# =========================
# AUDIO CONFIG
# =========================
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FIXED_FRAMES = 128   # makes 128x128 mel

# =========================
# MEL EXTRACTION
# =========================
def extract_mel(audio_path):
    y, sr = librosa.load(audio_path, sr=SR, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel = librosa.power_to_db(mel, ref=np.max)

    # normalize [0,1]
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)

    # pad / crop time axis
    if mel.shape[1] < FIXED_FRAMES:
        pad_width = FIXED_FRAMES - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad_width)))
    else:
        mel = mel[:, :FIXED_FRAMES]

    return mel.astype(np.float32)


# =========================
# PROCESS SPLIT
# =========================
def process_split(csv_path, split_name):
    print(f"\n🔄 Processing {split_name}")

    df = pd.read_csv(csv_path)

    X = []
    y = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = row["path"]
        label = row["label"]

        mel = extract_mel(audio_path)
        mel = np.expand_dims(mel, axis=-1)  # (128,128,1)

        X.append(mel)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    np.save(os.path.join(FEATURE_DIR, f"X_{split_name}.npy"), X)
    np.save(os.path.join(FEATURE_DIR, f"y_{split_name}.npy"), y)

    print(f"✅ Saved: X_{split_name}.npy | y_{split_name}.npy")
    print(f"   Shape X: {X.shape} | y: {y.shape}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    process_split(TRAIN_CSV, "train")
    process_split(VAL_CSV, "val")
    process_split(TEST_CSV, "test")

    print("\n🎉 Feature extraction COMPLETE")
