import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# ========= CONFIG =========
FILE_PATH = r"C:\Users\ADMIN\Downloads\music_dataset\vibraphone\100.wav"
# ==========================

# 1. Load audio
y, sr = librosa.load(FILE_PATH, sr=None, mono=True)
duration = len(y) / sr

print(f"File: {FILE_PATH}")
print(f"Sample rate: {sr} Hz")
print(f"Samples: {len(y)}")
print(f"Duration: {duration:.2f} sec")

# Common params
n_fft = 2048
hop = 512

# 2. STFT Spectrogram
stft = librosa.stft(y, n_fft=n_fft, hop_length=hop)
stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

# 3. Mel Spectrogram
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=128)
mel_db = librosa.amplitude_to_db(mel, ref=np.max)

# 4. MFCC
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=13)


# ========= PLOT EVERYTHING IN ONE TEMPLATE =========
plt.figure(figsize=(12, 14))

# -------- 1. Waveform --------
plt.subplot(4, 1, 1)
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# -------- 2. STFT Spectrogram --------
plt.subplot(4, 1, 2)
librosa.display.specshow(stft_db, sr=sr, hop_length=hop, x_axis="time", y_axis="hz")
plt.title("Spectrogram (STFT)")
plt.colorbar(format="%+2.0f dB")

# -------- 3. Mel Spectrogram --------
plt.subplot(4, 1, 3)
librosa.display.specshow(mel_db, sr=sr, hop_length=hop, x_axis="time", y_axis="mel")
plt.title("Mel Spectrogram")
plt.colorbar(format="%+2.0f dB")

# -------- 4. MFCC --------
plt.subplot(4, 1, 4)
librosa.display.specshow(mfcc, sr=sr, hop_length=hop, x_axis="time")
plt.title("MFCC")
plt.ylabel("MFCC Coefficient")
plt.colorbar()

plt.tight_layout()
plt.show()
