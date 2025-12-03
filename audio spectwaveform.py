import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load audio file
audio_path = r'C:\Users\manoj\Desktop\SchoolBoy_filtered_16khz.wav'
y, sr = librosa.load(audio_path, sr=None)   # keep original sample rate

# Create time axis
time = np.linspace(0, len(y) / sr, len(y))

# Create a figure with two subplots
plt.figure(figsize=(14, 8))

# ------------------ Waveform ------------------
plt.subplot(2, 1, 1)
plt.plot(time, y, color='blue')
plt.title('Waveform')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)

# ------------------ Spectrogram ------------------
D = librosa.stft(y)                                # STFT
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.subplot(2, 1, 2)
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
plt.title('Spectrogram (dB)')
plt.colorbar(format='%+2.0f dB')

plt.tight_layout()
plt.show()
