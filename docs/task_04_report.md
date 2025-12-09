# Task 04 Report: Preprocessing Pipeline & Dataset Preparation

## Overview
This task focused on implementing a robust preprocessing pipeline to prepare the IRMAS dataset for CNN training. The goal was to standardize the audio data to ensure consistent model input. Additionally, the Mel Spectrogram visualization was updated to use amplitude scaling instead of power scaling.

## Key Achievements

### 1. Preprocessing Pipeline Implementation
A comprehensive pipeline was developed in `src/audio_processor.py` and executed via `src/process_dataset.py`. The pipeline includes the following steps:
1.  **Loading:** Audio loaded using `librosa`.
2.  **Mono Conversion:** Stereo files converted to mono.
3.  **Resampling:** All audio resampled to **16 kHz**.
4.  **Normalization:** Amplitude normalized using `librosa.util.normalize`.
5.  **Silence Trimming:** Silence trimmed from beginning and end (`top_db=20`).
6.  **Duration Fixing:** Audio fixed to exactly **3 seconds** (48,000 samples).
    *   Shorter files: Padded with zeros.
    *   Longer files: Center cropped.

### 2. Dataset Processing
*   **Source:** `datasets/IRMAS-TrainingData`
*   **Destinations:** 
    *   Processed Audio: `outputs/processed_irmas` (WAV)
    *   Spectrograms: `outputs/mel_spectrograms_irmas` (PNG)
*   **Volume:** 6,705 files processed.
*   **Structure:** Maintained original class-based directory structure (e.g., `cel/`, `cla/`, etc.).

### 3. Visualization Updates
*   **Mel Spectrogram:** Updated `src/visualizer.py` to calculate Mel Spectrograms based on **Magnitude (Amplitude)** instead of Power.
    *   `librosa.feature.melspectrogram(..., power=1.0)`
    *   Converted to dB using `librosa.amplitude_to_db`.
    *   This ensures the visualization represents the log-amplitude correctly as requested.
*   **Batch Generation:** `src/process_dataset.py` now automatically generates and saves these spectrograms for every file in the dataset.

## Technical Details
*   **Target Sample Rate:** 16,000 Hz
*   **Target Duration:** 3.0 Seconds (48,000 Samples)
*   **Format:** WAV (Float32)

## Verification
A random sample check confirmed:
*   Sample Rate: 16000 Hz
*   Channels: 1 (Mono)
*   Duration: 3.0s

## Next Steps
*   Begin designing the CNN architecture.
*   Create data loaders (PyTorch/TensorFlow) to feed the processed `.wav` files (or generate spectrograms on-the-fly) into the model.
