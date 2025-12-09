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

def normalize_audio(y):
    """Normalizes the amplitude of the audio signal."""
    return librosa.util.normalize(y)

def trim_silence(y, top_db=20):
    """Trims silence from the beginning and end of the audio."""
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed

def fix_duration(y, sr, duration_seconds=3.0):
    """
    Fixes the duration of the audio to exactly duration_seconds.
    If shorter, pads with zeros. If longer, crops the center.
    """
    target_length = int(sr * duration_seconds)
    if len(y) < target_length:
        # Pad with zeros
        padding = target_length - len(y)
        y_fixed = np.pad(y, (0, padding), 'constant')
    elif len(y) > target_length:
        # Center crop
        start = (len(y) - target_length) // 2
        y_fixed = y[start : start + target_length]
    else:
        y_fixed = y
    
    return y_fixed

def process_audio_file(file_path, target_sr=16000, duration=3.0):
    """
    Loads an audio file, converts it to mono, resamples it, normalizes, 
    trims silence, and fixes duration.

    Args:
        file_path (str): The path to the audio file.
        target_sr (int): The target sampling rate for resampling.
        duration (float): Target duration in seconds.

    Returns:
        tuple: (processed_audio, sample_rate) or (None, None) if failure.
    """
    # Load audio
    try:
        y, sr = librosa.load(file_path, sr=None, mono=False)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

    # Mono conversion
    if y.ndim > 1:
        y = librosa.to_mono(y)

    # Resample
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    
    # Normalize
    y = normalize_audio(y)

    # Trim Silence
    y = trim_silence(y)

    # Fix Duration
    y = fix_duration(y, target_sr, duration)

    return y, target_sr

if __name__ == "__main__":
    pass
