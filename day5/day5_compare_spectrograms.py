import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Use cleaned audio from Day 4
BASE = "cleaned_audio"
INSTRUMENTS = {
    "guitar": "gac",
    "piano": "pia",
    "flute": "flu"
}

os.makedirs("outputs/day5", exist_ok=True)

def process_audio(inst_name, inst_code):
    # Pick first cleaned file
    folder = os.path.join(BASE, inst_code)
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    file_path = os.path.join(folder, files[0])

    print(f"\nProcessing {inst_name}: {file_path}")
    y, sr = librosa.load(file_path, sr=None)

    # --- Waveform ---
    plt.figure(figsize=(12, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"{inst_name.upper()} – Waveform")
    plt.tight_layout()
    plt.savefig(f"outputs/day5/{inst_name}_waveform.png")
    plt.close()

    # --- STFT ---
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure(figsize=(12, 3))
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="linear")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"{inst_name.upper()} – STFT Spectrogram")
    plt.tight_layout()
    plt.savefig(f"outputs/day5/{inst_name}_stft.png")
    plt.close()

    # --- Mel Spectrogram ---
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    plt.figure(figsize=(12, 3))
    librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"{inst_name.upper()} – Mel Spectrogram")
    plt.tight_layout()
    plt.savefig(f"outputs/day5/{inst_name}_mel.png")
    plt.close()

    # --- MFCC ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    plt.figure(figsize=(12, 3))
    librosa.display.specshow(mfcc, sr=sr, x_axis="time")
    plt.colorbar()
    plt.title(f"{inst_name.upper()} – MFCC")
    plt.tight_layout()
    plt.savefig(f"outputs/day5/{inst_name}_mfcc.png")
    plt.close()


def compare_visual(inst_name):
    """Create a side-by-side comparison figure."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Load saved spectrograms
    stft_img = plt.imread(f"outputs/day5/{inst_name}_stft.png")
    mel_img = plt.imread(f"outputs/day5/{inst_name}_mel.png")
    mfcc_img = plt.imread(f"outputs/day5/{inst_name}_mfcc.png")

    titles = ["STFT Spectrogram", "Mel Spectrogram", "MFCC"]

    for ax, img, title in zip(axes, [stft_img, mel_img, mfcc_img], titles):
        ax.imshow(img)
        ax.set_title(f"{inst_name.upper()} – {title}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"outputs/day5/{inst_name}_comparison.png")
    plt.close()


if __name__ == "__main__":
    for inst_name, inst_code in INSTRUMENTS.items():
        process_audio(inst_name, inst_code)
        compare_visual(inst_name)

    print("\nDay 5 processing complete! Check outputs/day5/")
