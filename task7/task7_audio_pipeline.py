import os
import librosa
import numpy as np

# Paths
BASE_AUDIO = "task4/cleaned_audio"
OUTPUT_DIR = "task7/outputs_task7"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INSTRUMENTS = {
    "guitar": "gac",
    "piano": "pia",
    "flute": "flu"
}

TARGET_DURATION = 3.0  # seconds
SR = 22050


def load_and_preprocess(path):
    """Load, trim silence, normalize, and fix duration"""
    y, sr = librosa.load(path, sr=SR, mono=True)

    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=20)

    # Fix duration
    target_len = int(TARGET_DURATION * SR)
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))

    # Normalize
    y = y / np.max(np.abs(y))

    return y, sr


def augment_audio(y):
    """Simple augmentation"""
    stretch = librosa.effects.time_stretch(y, rate=0.9)
    pitch = librosa.effects.pitch_shift(y=y, sr=SR, n_steps=2)
    return [y, stretch, pitch]


def extract_features(y, sr):
    """Extract MFCC features"""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)


def run_pipeline():
    for inst_name, inst_code in INSTRUMENTS.items():
        folder = os.path.join(BASE_AUDIO, inst_code)
        file = sorted([f for f in os.listdir(folder) if f.endswith(".wav")])[0]
        path = os.path.join(folder, file)

        print(f"Processing {inst_name}: {file}")

        y, sr = load_and_preprocess(path)
        augmented = augment_audio(y)

        features = []
        for audio in augmented:
            feat = extract_features(audio, sr)
            features.append(feat)

        features = np.array(features)
        np.save(f"{OUTPUT_DIR}/{inst_name}_features.npy", features)

    print("\nTask 7 data pipeline completed!")


if __name__ == "__main__":
    run_pipeline()
