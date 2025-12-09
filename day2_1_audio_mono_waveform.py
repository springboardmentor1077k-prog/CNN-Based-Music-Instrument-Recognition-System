import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Select any audio file from your dataset (change filename if needed)
audio_path = r"IRMAS-TrainingData/IRMAS-TrainingData/pia/[pia][cla]1283__1.wav"  # update to a valid filename

# Load without forcing mono
y, sr = librosa.load(audio_path, sr=None, mono=False)
print(f"Sample rate: {sr} Hz")

# Identify stereo or mono
if y.ndim == 1:
    print("Audio is already MONO")
    y_stereo = None
    y_mono = y
else:
    print("Audio is STEREO")
    y_stereo = y
    y_mono = librosa.to_mono(y)
    print("Converted to MONO successfully")

# Plot stereo (only if stereo)
if y_stereo is not None:
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y_stereo[0], sr=sr)
    plt.title("Waveform – Left Channel (Stereo)")
    plt.tight_layout()
    plt.savefig("outputs/day2_1_waveform_stereo_left.png")
    plt.close()

    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y_stereo[1], sr=sr)
    plt.title("Waveform – Right Channel (Stereo)")
    plt.tight_layout()
    plt.savefig("outputs/day2_1_waveform_stereo_right.png")
    plt.close()

# Plot mono waveform
plt.figure(figsize=(10, 3))
librosa.display.waveshow(y_mono, sr=sr)
plt.title("Waveform – MONO")
plt.tight_layout()
plt.savefig("outputs/day2_1_waveform_mono.png")
plt.close()

print("Saved waveform images in outputs/")
