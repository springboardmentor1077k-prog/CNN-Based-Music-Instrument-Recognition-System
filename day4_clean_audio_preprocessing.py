import os
import glob

import numpy as np
import librosa
import soundfile as sf  # pip install soundfile if not installed

# Base folder where your raw IRMAS audio is stored
RAW_BASE = "IRMAS-TrainingData/IRMAS-TrainingData"

# Where to save cleaned audio
CLEAN_BASE = "cleaned_audio"

# Instruments/folders you want to process (you can add more later)
INSTRUMENT_DIRS = ["gac", "pia", "flu"]  # guitar, piano, flute

# Target settings
TARGET_SR = 22050           # target sample rate
TARGET_DURATION = 3.0       # seconds
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)

def clean_single_file(input_path, output_path):
    # 1. Load audio (keep original, then convert)
    y, sr = librosa.load(input_path, sr=None, mono=False)
    print(f"\nProcessing: {input_path}")
    print(f"Original sr={sr}, shape={y.shape}")

    # 2. Convert to mono and resample to TARGET_SR
    if y.ndim > 1:
        y = librosa.to_mono(y)
        print("Converted from stereo to mono.")
    else:
        print("Audio is already mono.")

    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        print(f"Resampled from {sr} Hz to {TARGET_SR} Hz.")
    sr = TARGET_SR

    # 3. Trim leading/trailing silence
    y_trimmed, idx = librosa.effects.trim(y, top_db=20)
    start_sec = idx[0] / sr
    end_sec = idx[1] / sr
    print(f"Trimmed silence: kept from {start_sec:.2f}s to {end_sec:.2f}s")

    # 4. Fix duration (pad or cut to TARGET_DURATION)
    cur_len = len(y_trimmed)
    if cur_len > TARGET_SAMPLES:
        y_fixed = y_trimmed[:TARGET_SAMPLES]
        print(f"Clipped from {cur_len/sr:.2f}s to {TARGET_DURATION:.2f}s")
    else:
        pad_len = TARGET_SAMPLES - cur_len
        y_fixed = np.pad(y_trimmed, (0, pad_len), mode="constant")
        print(f"Padded from {cur_len/sr:.2f}s to {TARGET_DURATION:.2f}s")

    # 5. Save clean audio
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, y_fixed, sr)
    print(f"Saved cleaned audio to: {output_path}")

def process_instruments():
    for inst_code in INSTRUMENT_DIRS:
        input_folder = os.path.join(RAW_BASE, inst_code)
        output_folder = os.path.join(CLEAN_BASE, inst_code)

        pattern = os.path.join(input_folder, "*.wav")
        files = glob.glob(pattern)

        if not files:
            print(f"No files found in {input_folder}")
            continue

        print(f"\n=== Processing instrument: {inst_code} ({len(files)} files) ===")

        # For now, process only first 1 files per instrument to test
        for path in files[:1]:
            filename = os.path.basename(path)
            out_path = os.path.join(output_folder, filename.replace(".wav", "_clean.wav"))
            clean_single_file(path, out_path)

if __name__ == "__main__":
    process_instruments()
    print("\nDone cleaning audio for selected instruments.")
