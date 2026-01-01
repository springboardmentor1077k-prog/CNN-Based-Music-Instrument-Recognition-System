# Task 1: Audio Exploration and Setup Report

**Date:** December 2, 2025
**Status:** Completed

## 1. Objectives
The primary goals of this task were to:
- Explore the IRMAS dataset structure and audio properties.
- Implement a Python script to load and process audio.
- Standardize audio format (Mono, 16kHz).
- Visualize the audio waveform.

## 2. Implementation
A Python script `src/audio_processor.py` was created with the following features:

### Dataset Exploration
An `explore_dataset` function was added to traverse `datasets/IRMAS-TrainingData`.
- **Total Files:** 6705
- **Class Distribution:**
  - `cel`: 388
  - `cla`: 505
  - `flu`: 451
  - `gac`: 637
  - `gel`: 760
  - `org`: 682
  - `pia`: 721
  - `sax`: 626
  - `tru`: 577
  - `vio`: 580
  - `voi`: 778

### Audio Processing Pipeline
The script `process_audio_file` performs:
1.  **Loading:** Uses `librosa` to load audio.
2.  **Stereo Check:** Detects if `ndim > 1`.
3.  **Mono Conversion:** Uses `librosa.to_mono` if necessary.
4.  **Resampling:** Resamples to 16,000 Hz using `librosa.resample`.
5.  **Visualization:** Generates and saves a waveform plot using `matplotlib`.

## 3. Findings & Data Analysis
- **Source Format:** The IRMAS training files are predominantly **Stereo (2 channels)** with a sample rate of **44.1 kHz**.
- **Duration:** Files are typically ~3 seconds long.
- **Dependencies:** The project successfully utilizes `librosa`, `numpy`, and `matplotlib`.

## 4. Outputs
- **Script:** `src/audio_processor.py` is fully functional.
- **Artifacts:** A sample waveform image `waveform_example.png` was generated in the project root.
