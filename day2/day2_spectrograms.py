import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output folder if not exist
os.makedirs("outputs", exist_ok=True)

# 1. Load an audio file
audio_path = r"C:\Users\hp\Desktop\Hechie-techiE\Infosys 6.0\CNN-Based-Music-Instrument-Recognition-System\IRMAS-TrainingData\IRMAS-TrainingData\org\[org][pop_roc]1259__2.wav"

y, sr = librosa.load(audio_path, sr=None)

# Waveform
plt.figure(figsize=(12, 3))
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform - ORG")
plt.tight_layout()
plt.savefig("outputs/waveform_org.png")
plt.close()

# STFT Spectrogram
D = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure(figsize=(12, 3))
librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="linear")
plt.title("Spectrogram (STFT, linear) - ORG")
plt.colorbar(format="%+2.0f dB")
plt.tight_layout()
plt.savefig("outputs/spectrogram_org.png")
plt.close()

# Mel-Spectrogram
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

plt.figure(figsize=(12, 3))
librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel")
plt.title("Mel-Spectrogram - ORG")
plt.colorbar(format="%+2.0f dB")
plt.tight_layout()
plt.savefig("outputs/mel_spectrogram_org.png")
plt.close()

# MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

plt.figure(figsize=(12, 3))
librosa.display.specshow(mfccs, sr=sr, x_axis="time")
plt.title("MFCCs - ORG")
plt.xlabel("Time (s)")
plt.ylabel("MFCC index")
plt.colorbar()
plt.tight_layout()
plt.savefig("outputs/mfcc_org.png")
plt.close()

print("All plots saved successfully in outputs/ folder!")

# ==============================
# Combined comparison figure
# ==============================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 6))

# Waveform
librosa.display.waveshow(y, sr=sr, ax=axes[0, 0])
axes[0, 0].set_title("Waveform")

# Linear Spectrogram
img1 = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="linear", ax=axes[0, 1])
axes[0, 1].set_title("Spectrogram (STFT, linear)")

# Mel-Spectrogram
img2 = librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel", ax=axes[1, 0])
axes[1, 0].set_title("Mel-Spectrogram")

# MFCCs
img3 = librosa.display.specshow(mfccs, sr=sr, x_axis="time", ax=axes[1, 1])
axes[1, 1].set_title("MFCCs")

fig.colorbar(img1, ax=axes[0, 1], format="%+2.0f dB")
fig.colorbar(img2, ax=axes[1, 0], format="%+2.0f dB")
fig.colorbar(img3, ax=axes[1, 1])

plt.tight_layout()
plt.savefig("outputs/comparison_org.png")
plt.close()

print("Saved comparison figure: outputs/comparison_org.png")

