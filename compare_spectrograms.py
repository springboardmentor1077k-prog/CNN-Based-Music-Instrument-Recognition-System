import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
# We will pick one random file from your IRMAS dataset to test
DATASET_PATH = './IRMAS-TrainingData/IRMAS-TrainingData' 

def compare_visuals():
    # 1. Find a random file
    if not os.path.exists(DATASET_PATH):
        print("Error: Dataset path not found.")
        return

    # Get a list of all wav files in the 'pia' (Piano) folder for a clear example
    piano_path = os.path.join(DATASET_PATH, 'flu')
    if not os.path.exists(piano_path):
        # Fallback if 'pia' doesn't exist, just grab the first available folder
        piano_path = os.path.join(DATASET_PATH, os.listdir(DATASET_PATH)[0])
    
    files = [f for f in os.listdir(piano_path) if f.endswith('.wav')]
    sample_file = os.path.join(piano_path, files[0])
    
    print(f"Analyzing: {files[0]}")

    # 2. Load Audio
    y, sr = librosa.load(sample_file, sr=16000, mono=True)

    # --- A. GENERATE NORMAL (LINEAR) SPECTROGRAM ---
    # We use Short-Time Fourier Transform (STFT)
    D = librosa.stft(y)
    # Convert to decibels so we can see it
    linear_spectrogram = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # --- B. GENERATE MEL SPECTROGRAM ---
    # We use the Mel filter bank
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    # Convert to decibels
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # --- C. PLOT THEM SIDE-BY-SIDE ---
    plt.figure(figsize=(14, 6))
   # plt.title('Spectrogram Comparison: PIANO')

    # Plot 1: Linear
    plt.subplot(1, 2, 1)
    librosa.display.specshow(linear_spectrogram, sr=sr, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('1. Normal Spectrogram(STFT)')
    plt.xlabel('Time')
    plt.ylabel('Frequency (Hz)')

    # Plot 2: Mel
    plt.subplot(1, 2, 2)
    librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('2. Mel-Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency (Mel)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_visuals()