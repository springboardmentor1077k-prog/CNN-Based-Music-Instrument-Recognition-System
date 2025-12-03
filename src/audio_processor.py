import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# Define paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "IRMAS-TrainingData")

def explore_dataset(dataset_path):
    """
    Explores the IRMAS dataset directory structure and prints statistics.
    """
    print(f"--- Exploring Dataset at {dataset_path} ---")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory not found at {dataset_path}")
        return

    class_counts = {}
    total_files = 0
    
    # Walk through the directory
    for root, dirs, files in os.walk(dataset_path):
        # Skip the root folder itself, only look at subfolders (classes)
        if root == dataset_path:
            continue
            
        class_name = os.path.basename(root)
        wav_files = [f for f in files if f.lower().endswith('.wav')]
        count = len(wav_files)
        
        if count > 0:
            class_counts[class_name] = count
            total_files += count

    print(f"Total audio files found: {total_files}")
    print("Files per class:")
    for cls, count in class_counts.items():
        print(f"  - {cls}: {count}")

    # Check a few random files for properties
    print("\n--- Checking Random File Properties ---")
    all_wavs = []
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith('.wav'):
                all_wavs.append(os.path.join(root, f))

    if all_wavs:
        sample_files = random.sample(all_wavs, min(3, len(all_wavs)))
        for file_path in sample_files:
            try:
                # Use librosa to get info without fully loading data if possible, 
                # but librosa.load is standard. get_duration is fast.
                # For channels/sr, we usually load.
                y, sr = librosa.load(file_path, sr=None, mono=False)
                duration = librosa.get_duration(y=y, sr=sr)
                channels = y.shape[0] if y.ndim > 1 else 1
                print(f"File: {os.path.basename(file_path)}")
                print(f"  - Duration: {duration:.2f}s, Sample Rate: {sr}Hz, Channels: {channels}")
            except Exception as e:
                print(f"  - Error reading {os.path.basename(file_path)}: {e}")
    else:
        print("No WAV files found to check.")
    print("---------------------------------------\n")

def process_audio_file(file_path, target_sr=16000):
    """
    Loads an audio file, converts it to mono if stereo, resamples it to a target sample rate,
    and generates a waveform plot.

    Args:
        file_path (str): The path to the audio file.
        target_sr (int): The target sampling rate for resampling.

    Returns:
        tuple: A tuple containing the processed audio (numpy array) and its sampling rate (int).
    """
    print(f"Processing: {file_path}")
    # Load audio
    try:
        y, sr = librosa.load(file_path, sr=None, mono=False)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None

    # Identify stereo and convert to mono if necessary
    if y.ndim > 1:
        print(f"  - Original audio is stereo (channels: {y.shape[0]}). Converting to mono.")
        y_mono = librosa.to_mono(y)
    else:
        print("  - Original audio is mono.")
        y_mono = y

    # Resample to target_sr
    if sr != target_sr:
        print(f"  - Original sampling rate: {sr} Hz. Resampling to {target_sr} Hz.")
        y_resampled = librosa.resample(y_mono, orig_sr=sr, target_sr=target_sr)
        sr_processed = target_sr
    else:
        print(f"  - Original sampling rate: {sr} Hz. No resampling needed.")
        y_resampled = y_mono
        sr_processed = sr

    # Generate waveform graph (placeholder for now, will be enhanced)
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y_resampled, sr=sr_processed)
    plt.title(f'Waveform of {os.path.basename(file_path)} (Processed)')
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    # plt.savefig(f"waveform_{os.path.basename(file_path)}.png") # Uncomment to save
    # plt.show() # Uncomment to display
    print(f"  - Audio processed. Shape: {y_resampled.shape}, Sample Rate: {sr_processed} Hz")

    return y_resampled, sr_processed

if __name__ == "__main__":
    # 1. Explore the dataset
    explore_dataset(DATASET_DIR)

    # 2. Execute processing on a sample file
    print("Executing sample processing task...")
    
    # Construct path to a specific file we know exists (or should exist)
    # Based on file listing: datasets/IRMAS-TrainingData/cel/[cel][cla]0001__1.wav
    test_audio_file_path = os.path.join(DATASET_DIR, "cel", "[cel][cla]0001__1.wav")
    
    # Ensure outputs directory exists
    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_image_path = os.path.join(output_dir, "waveform_example.png") 

    if os.path.exists(test_audio_file_path):
        processed_audio, processed_sr = process_audio_file(test_audio_file_path)
        if processed_audio is not None:
            plt.savefig(output_image_path)
            print(f"  - Waveform plot saved to {output_image_path}")
            plt.close() 
    else:
        print(f"Test file not found: {test_audio_file_path}")
        print("Please ensure the IRMAS dataset is unpacked in 'datasets/IRMAS-TrainingData'.")
