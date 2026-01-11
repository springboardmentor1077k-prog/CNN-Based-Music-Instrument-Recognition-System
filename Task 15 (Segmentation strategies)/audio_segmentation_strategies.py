# =====================================================
# Audio Segmentation Strategies (IRMAS)
# Test with 5 Random Segments
# =====================================================

import os
import random
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# CONFIGURATION
# =====================================================

DATASET_ROOT = "IRMAS-TrainingData"

SAMPLE_RATE = 22050
SEGMENT_DURATION = 1.0        # seconds per segment
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION)

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

NUM_TEST_SEGMENTS = 5

# =====================================================
# AUDIO LOADING
# =====================================================

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return y, sr

# =====================================================
# SEGMENTATION STRATEGIES
# =====================================================

def random_segmentation(y, segment_samples, num_segments):
    """Randomly sample segments from the audio"""
    segments = []
    max_start = len(y) - segment_samples

    for _ in range(num_segments):
        start = random.randint(0, max_start)
        segment = y[start:start + segment_samples]
        segments.append(segment)

    return segments

def fixed_segmentation(y, segment_samples):
    """Split audio into fixed non-overlapping segments"""
    segments = []
    for start in range(0, len(y) - segment_samples + 1, segment_samples):
        segment = y[start:start + segment_samples]
        segments.append(segment)
    return segments

def overlapping_segmentation(y, segment_samples, overlap=0.5):
    """Split audio into overlapping segments"""
    hop = int(segment_samples * (1 - overlap))
    segments = []
    for start in range(0, len(y) - segment_samples + 1, hop):
        segment = y[start:start + segment_samples]
        segments.append(segment)
    return segments

# =====================================================
# MEL-SPECTROGRAM
# =====================================================

def mel_spectrogram(y, sr):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    return librosa.power_to_db(mel, ref=np.max)

# =====================================================
# VISUALIZE SEGMENTS
# =====================================================

def visualize_segments(segments, sr, title):
    plt.figure(figsize=(12, 8))

    for i, segment in enumerate(segments, 1):
        mel_db = mel_spectrogram(segment, sr)
        plt.subplot(NUM_TEST_SEGMENTS, 1, i)
        librosa.display.specshow(
            mel_db,
            sr=sr,
            x_axis="time",
            y_axis="mel"
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"{title} – Segment {i}")

    plt.tight_layout()
    plt.show()

# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":

    print("Starting segmentation strategy testing")

    # Pick ONE random IRMAS audio file
    all_wavs = []
    for root, _, files in os.walk(DATASET_ROOT):
        for file in files:
            if file.lower().endswith(".wav"):
                all_wavs.append(os.path.join(root, file))

    audio_path = random.choice(all_wavs)
    print("Selected audio:", audio_path)

    y, sr = load_audio(audio_path)

    # ---------- Strategy 1: Random Segments ----------
    random_segments = random_segmentation(
        y,
        SEGMENT_SAMPLES,
        NUM_TEST_SEGMENTS
    )
    visualize_segments(random_segments, sr, "Random Segmentation")

    # ---------- Strategy 2: Fixed Segments ----------
    fixed_segments = fixed_segmentation(y, SEGMENT_SAMPLES)
    visualize_segments(
        fixed_segments[:NUM_TEST_SEGMENTS],
        sr,
        "Fixed Segmentation"
    )

    # ---------- Strategy 3: Overlapping Segments ----------
    overlap_segments = overlapping_segmentation(
        y,
        SEGMENT_SAMPLES,
        overlap=0.5
    )
    visualize_segments(
        overlap_segments[:NUM_TEST_SEGMENTS],
        sr,
        "Overlapping Segmentation (50%)"
    )

    print("Segmentation testing completed")
