import os
import librosa
import soundfile as sf
import numpy as np

# ========== CONFIG ==========
SOURCE_TRAIN = os.path.join("task4_nsynthmini_preprocessed", "train")
OUT_ROOT = "augmented"
OUT_TRAIN = os.path.join(OUT_ROOT, "train")

N_EXAMPLES = 3   # take 3 training files
TARGET_SR = 60000
# ===========================

os.makedirs(OUT_TRAIN, exist_ok=True)


# --- Augmentation Functions ---
def add_noise(y, snr_db=20):
    sig_power = np.mean(y ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(len(y))
    return (y + noise).astype("float32")


def time_shift(y, sr, shift_seconds):
    samples = int(shift_seconds * sr)
    return np.roll(y, samples).astype("float32")


def pitch_shift(y, sr, n_steps):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps).astype("float32")


def time_stretch(y, rate):
    return librosa.effects.time_stretch(y, rate=rate).astype("float32")


def change_gain(y, gain):
    return (y * gain).astype("float32")


def safe_write(path, y, sr):
    max_val = np.max(np.abs(y))
    if max_val > 1.0:
        y = y / max_val
    sf.write(path, y, sr)


# --- Processing Function ---
def augment_file(path, out_folder, sr=TARGET_SR):
    basename = os.path.splitext(os.path.basename(path))[0]
    y, _ = librosa.load(path, sr=sr, mono=True)

    outputs = []

    # 1) Time Stretch
    for rate in [0.9, 1.1]:
        out = os.path.join(out_folder, f"{basename}__aug-tstretch-{rate}.wav")
        safe_write(out, time_stretch(y, rate), sr)
        outputs.append(out)

    # 2) Pitch Shift
    for steps in [2, -2]:
        out = os.path.join(out_folder, f"{basename}__aug-pitch-{steps:+d}.wav")
        safe_write(out, pitch_shift(y, sr, steps), sr)
        outputs.append(out)

    # 3) Noise
    out = os.path.join(out_folder, f"{basename}__aug-noise.wav")
    safe_write(out, add_noise(y, snr_db=20), sr)
    outputs.append(out)

    # 4) Time Shift
    for shift in [0.2, -0.2]:
        direction = "pos" if shift > 0 else "neg"
        out = os.path.join(out_folder, f"{basename}__aug-timeshift-{direction}.wav")
        safe_write(out, time_shift(y, sr, shift), sr)
        outputs.append(out)

    # 5) Gain
    for g in [0.7, 1.3]:
        out = os.path.join(out_folder, f"{basename}__aug-gain-{g}.wav")
        safe_write(out, change_gain(y, g), sr)
        outputs.append(out)

    return outputs


# --- MAIN ---
def main():
    print("\nScanning training folder:", SOURCE_TRAIN)

    files = [f for f in os.listdir(SOURCE_TRAIN) if f.endswith(".wav")]
    files.sort()

    if len(files) == 0:
        print("No WAV files found.")
        return

    selected = files[:N_EXAMPLES]
    print("\nSelected for augmentation:", selected)

    all_outputs = []

    for fname in selected:
        full_path = os.path.join(SOURCE_TRAIN, fname)
        print(f"\nAugmenting {fname}...")
        outputs = augment_file(full_path, OUT_TRAIN)
        for o in outputs:
            print(" Saved:", o)
        all_outputs.extend(outputs)

    print("\n====================================")
    print("AUGMENTATION COMPLETE")
    print(f"Saved {len(all_outputs)} augmented files in:")
    print(f"  {OUT_TRAIN}")
    print("====================================\n")


if __name__ == "__main__":
    main()
