import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
# UPDATE THIS LINE
DATASET_PATH = './IRMAS-TrainingData/IRMAS-TrainingData'  # Path to your dataset
TARGET_SR = 16000                      # 16kHz sampling rate

def process_dataset():
    # 1. Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Folder '{DATASET_PATH}' not found. Check your file structure!")
        return

    print(f"Scanning '{DATASET_PATH}'...\n")
    
    # Get list of all instrument folders (pia, gel, cel, etc.)
    instruments = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    
    print(f"Found {len(instruments)} instrument classes: {instruments}\n")

    # Let's process ONE file from the first instrument folder to test
    first_instrument = instruments[0]
    instrument_folder = os.path.join(DATASET_PATH, first_instrument)
    
    # Find the first .wav file in that folder
    audio_files = [f for f in os.listdir(instrument_folder) if f.endswith('.wav')]
    
    if not audio_files:
        print(f"No .wav files found in {first_instrument} folder.")
        return

    sample_file_path = os.path.join(instrument_folder, audio_files[0])
    
    # --- STEP A: IDENTIFY STEREO VS MONO ---
    # We load with sr=None to see the ORIGINAL properties first
    y_orig, sr_orig = librosa.load(sample_file_path, sr=None, mono=False)
    
    print(f"--- File: {audio_files[0]} ---")
    print(f"Original Sample Rate: {sr_orig} Hz")
    
    # Check shape: (Channels, Samples)
    if y_orig.ndim > 1:
        print(f"Original Audio is STEREO (Channels: {y_orig.shape[0]})")
    else:
        print("Original Audio is MONO")

    # --- STEP B: CONVERT TO MONO & 16KHZ ---
    print("\nProcessing... (Converting to Mono & 16kHz)")
    # Librosa handles the math: averages channels (mono) and resamples (sr)
    y_processed, sr_new = librosa.load(sample_file_path, sr=TARGET_SR, mono=True)

    print(f"New Sample Rate: {sr_new} Hz")
    print(f"New Shape: {y_processed.shape} (1D array = Mono)")

    # --- STEP C: PLOT WAVEFORM ---
    plt.figure(figsize=(12, 6))
    
    # Plot Original (just one channel if stereo)
    plt.subplot(2, 1, 1)
    if y_orig.ndim > 1:
        librosa.display.waveshow(y_orig[0], sr=sr_orig, color='blue', alpha=0.5)
        plt.title(f"Original (Stereo - Left Channel) @ {sr_orig}Hz")
    else:
        librosa.display.waveshow(y_orig, sr=sr_orig, color='blue', alpha=0.5)
        plt.title(f"Original (Mono) @ {sr_orig}Hz")
        
    # Plot Processed
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(y_processed, sr=sr_new, color='green', alpha=0.5)
    plt.title(f"Processed (Mono) @ {sr_new}Hz")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_dataset()