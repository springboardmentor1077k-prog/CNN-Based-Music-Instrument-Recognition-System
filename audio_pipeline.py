import os
import librosa
import numpy as np
import soundfile as sf
import pandas as pd
from tqdm import tqdm
import librosa.display
import matplotlib.pyplot as plt
import random

# ============================
# CONFIGURATION
# ============================

RAW_DATA = "nsynth_wav"          # Input dataset from extractor
OUTPUT_DIR = "task4"             # Output dataset root
TARGET_SR = 60000                # 60 kHz sample rate
FIX_DURATION = 4.0               # 4 seconds fixed duration
NUM_SAMPLES = int(TARGET_SR * FIX_DURATION)

AUG_COUNT = 3  # number of augmentations per audio file (train only)

# Make output folders
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/audio_clean", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/audio_augmented", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/mel", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/metadata", exist_ok=True)


# ============================
# AUDIO CLEANING FUNCTION
# ============================

def preprocess_audio(path):
    """Load → mono → resample → trim → normalize → fix duration."""
    y, sr = librosa.load(path, sr=None, mono=False)

    # Stereo → mono
    if y.ndim > 1:
        y = librosa.to_mono(y)

    # Resample to target sr
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=30)

    # Normalize
    y = y / (np.max(np.abs(y)) + 1e-9)

    # Fix 4-second duration
    if len(y) > NUM_SAMPLES:
        y = y[:NUM_SAMPLES]
    else:
        y = np.pad(y, (0, NUM_SAMPLES - len(y)), mode="constant")

    return y


# ============================
# AUGMENTATION FUNCTION (SAFE FOR ALL LIBROSA VERSIONS)
# ============================

def augment_audio(y):
    """Generate augmented versions using pitch shift, resample-stretch, noise."""

    augmented = []

    # 1. Pitch Shift
    n_steps = random.choice([-2, -1, 1, 2])
    y_pitch = librosa.effects.pitch_shift(y, sr=TARGET_SR, n_steps=n_steps)
    augmented.append(y_pitch)

    # 2. Time Stretch using Safe Resample Trick
    rate = random.choice([0.9, 1.1])  # slow down or speed up

    # Stretch by resampling → works for all livrosa versions
    stretched_sr = int(TARGET_SR * rate)
    y_stretch = librosa.resample(y, orig_sr=TARGET_SR, target_sr=stretched_sr)

    # Fix duration back to 4 sec
    if len(y_stretch) > NUM_SAMPLES:
        y_stretch = y_stretch[:NUM_SAMPLES]
    else:
        y_stretch = np.pad(y_stretch, (0, NUM_SAMPLES - len(y_stretch)), mode="constant")

    augmented.append(y_stretch)

    # 3. Add Gaussian Noise
    noise = np.random.normal(0, 0.01, len(y))
    y_noise = y + noise
    y_noise = y_noise / (np.max(np.abs(y_noise)) + 1e-9)
    augmented.append(y_noise)

    return augmented[:AUG_COUNT]


# ============================
# MEL SPECTROGRAM FUNCTION
# ============================

def save_mel_spectrogram(y, sr, save_path):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(3, 3))
    librosa.display.specshow(mel_db, sr=sr, hop_length=512)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


# ============================
# PROCESS EACH SPLIT
# ============================

def process_split(split):
    print(f"\nProcessing {split}...")

    split_in = os.path.join(RAW_DATA, split)
    split_clean = os.path.join(OUTPUT_DIR, "audio_clean", split)
    split_aug = os.path.join(OUTPUT_DIR, "audio_augmented", split)
    split_mel = os.path.join(OUTPUT_DIR, "mel", split)

    os.makedirs(split_clean, exist_ok=True)
    os.makedirs(split_mel, exist_ok=True)
    if split == "train":
        os.makedirs(split_aug, exist_ok=True)

    rows = []

    for file in tqdm(os.listdir(split_in)):
        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(split_in, file)

        # --- CLEAN AUDIO ---
        y = preprocess_audio(file_path)

        clean_path = os.path.join(split_clean, file)
        sf.write(clean_path, y, TARGET_SR)

        mel_path = os.path.join(split_mel, file.replace(".wav", ".png"))
        save_mel_spectrogram(y, TARGET_SR, mel_path)

        rows.append([file, clean_path, mel_path, "clean"])

        # --- AUGMENTATION (TRAIN ONLY) ---
        if split == "train":
            aug_versions = augment_audio(y)

            for i, aug_y in enumerate(aug_versions):
                aug_file = file.replace(".wav", f"_aug{i+1}.wav")
                aug_path = os.path.join(split_aug, aug_file)

                sf.write(aug_path, aug_y, TARGET_SR)

                aug_mel = os.path.join(split_mel, aug_file.replace(".wav", ".png"))
                save_mel_spectrogram(aug_y, TARGET_SR, aug_mel)

                rows.append([aug_file, aug_path, aug_mel, "augmented"])

    # Save metadata CSV
    df = pd.DataFrame(rows, columns=["filename", "audio_path", "mel_path", "type"])
    df.to_csv(f"{OUTPUT_DIR}/metadata/{split}.csv", index=False)

    print(f"✔ {split.upper()} completed with {len(rows)} items.")


# ============================
# RUN THE PIPELINE
# ============================

for split in ["train", "valid", "test"]:
    process_split(split)

print("\n🎉 AUDIO PIPELINE COMPLETED SUCCESSFULLY!")
print("Processed dataset saved in: task4/")
