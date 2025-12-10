import os
import glob
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Use cleaned audio from Day 4
BASE = "day4/cleaned_audio"
INSTRUMENTS = {
    "guitar": "gac",
    "piano": "pia",
    "flute": "flu"
}

OUT_DIR = "outputs/day5"
os.makedirs(OUT_DIR, exist_ok=True)

# OPTIONAL: remove old non-comparison images to keep outputs folder clean
for f in glob.glob(os.path.join(OUT_DIR, "*")):
    if "comparison" not in os.path.basename(f).lower():
        try:
            os.remove(f)
        except Exception:
            pass

def create_comparison(inst_name, inst_code):
    folder = os.path.join(BASE, inst_code)
    files = sorted([f for f in os.listdir(folder) if f.endswith(".wav")])
    if not files:
        print(f"WARNING: no .wav files found for {inst_name} in {folder}")
        return

    # pick the first cleaned file (change if you want a specific file)
    file_path = os.path.join(folder, files[0])
    print(f"Processing {inst_name}: {file_path}")

    y, sr = librosa.load(file_path, sr=None)

    # Parameters for STFT / mel
    n_fft = 2048
    hop_length = 512
    win_length = None  # default

    # STFT (linear-frequency spectrogram)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Mel spectrogram (power -> dB)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128, fmax=sr//2)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # MFCC (from mel)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=13)

    # Create a single comparison figure (3 rows)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)

    # STFT
    ax = axes[0]
    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="linear", ax=ax)
    ax.set_title(f"{inst_name.upper()} — STFT Spectrogram (dB, linear freq)")
    ax.set_ylabel("Hz")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    # Mel
    ax = axes[1]
    img2 = librosa.display.specshow(mel_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title(f"{inst_name.upper()} — Mel Spectrogram (dB, Mel scale)")
    ax.set_ylabel("Mel bins")
    fig.colorbar(img2, ax=ax, format="%+2.0f dB")

    # MFCC
    ax = axes[2]
    img3 = librosa.display.specshow(mfcc, sr=sr, hop_length=hop_length, x_axis="time", ax=ax)
    ax.set_title(f"{inst_name.upper()} — MFCC (n_mfcc=13)")
    ax.set_ylabel("MFCC")
    ax.set_xlabel("Time (s)")
    fig.colorbar(img3, ax=ax)

    # Save ONLY the comparison image
    out_path = os.path.join(OUT_DIR, f"{inst_name}_comparison.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved comparison: {out_path}")

if __name__ == "__main__":
    for inst_name, inst_code in INSTRUMENTS.items():
        create_comparison(inst_name, inst_code)

    print("\nDay 5 processing complete! Check", OUT_DIR)
