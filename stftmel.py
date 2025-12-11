import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# --- CONFIGURATION ---
DATASET_PATH = './IRMAS-TrainingData/IRMAS-TrainingData' 

def compare_visuals():
    if not os.path.exists(DATASET_PATH):
        print("Error: Dataset path not found.")
        return


    instrument_code = 'flu' 
    instrument_folder = os.path.join(DATASET_PATH, instrument_code)
    
    if not os.path.exists(instrument_folder):
        # Fallback if folder doesn't exist
        instrument_folder = os.path.join(DATASET_PATH, os.listdir(DATASET_PATH)[0])
    
    files = [f for f in os.listdir(instrument_folder) if f.endswith('.wav')]
    
    
    sample_file = os.path.join(instrument_folder, files[0])
    
    print(f"Analyzing: {sample_file}")

    # Load Audio
    y, sr = librosa.load(sample_file, sr=16000, mono=True)

    # Generate Linear Spectrogram
    D = librosa.stft(y)
    linear_spectrogram = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Generate Mel Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # --- PLOTTING ---
    plt.figure(figsize=(14, 6))

    # 1. Add the Main Title (Super Title)
    # We use .upper() to make 'pia' look like 'PIANO' (or you can hardcode it)
    full_title = f"Spectrogram Comparison: {instrument_code.upper()}" 
    plt.suptitle(full_title, fontsize=16, fontweight='bold')

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

    # 2. Fix the Alignment
    # rect=[left, bottom, right, top]
    # top=0.90 means "stop the graphs at 90% height", leaving 10% for the title
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    
    plt.show()

if __name__ == "__main__":
    compare_visuals()