import os
import librosa
import numpy as np
import soundfile as sf

BASE = "../task4/cleaned_audio"
OUT = "augmented_audio"

INSTRUMENTS = {
    "guitar": "gac",
    "piano": "pia",
    "flute": "flu"
}

os.makedirs(OUT, exist_ok=True)

def augment_audio(inst_name, inst_code):
    input_dir = os.path.join(BASE, inst_code)
    output_dir = os.path.join(OUT, inst_name)
    os.makedirs(output_dir, exist_ok=True)

    # Pick one cleaned audio file
    file = [f for f in os.listdir(input_dir) if f.endswith(".wav")][0]
    path = os.path.join(input_dir, file)

    y, sr = librosa.load(path, sr=None)

    # 1️⃣ Time Stretch
    stretch = librosa.effects.time_stretch(y, rate=0.9)

    # 2️⃣ Pitch Shift
    pitch = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2)

    # 3️⃣ Add Noise
    noise = y + 0.005 * np.random.randn(len(y))

    sf.write(f"{output_dir}/{inst_name}_stretch.wav", stretch, sr)
    sf.write(f"{output_dir}/{inst_name}_pitch.wav", pitch, sr)
    sf.write(f"{output_dir}/{inst_name}_noise.wav", noise, sr)

    print(f"Augmented audio saved for {inst_name}")

if __name__ == "__main__":
    for inst, code in INSTRUMENTS.items():
        augment_audio(inst, code)

    print("\nTask 6 audio augmentation completed!")
