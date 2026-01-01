# Task 5 Report: Instrument Spectrogram Comparison

**Date:** December 9, 2025

## Objective
The primary objective of Task 5 was to visually compare the spectral characteristics of various musical instruments using STFT and Mel-spectrograms. This involved selecting one audio sample for each instrument class from the IRMAS dataset, processing these samples, and generating a combined plot for side-by-side comparison.

## Methodology

1.  **Instrument Selection:** A representative `.wav` file was selected for each of the 11 instrument classes present in the IRMAS dataset (Cello, Clarinet, Flute, Acoustic Guitar, Electric Guitar, Organ, Piano, Saxophone, Trumpet, Violin, Voice). The file paths were hardcoded into the comparison script for consistency.

2.  **Audio Processing:** The `src/audio_processor.py` module's `process_audio_file` function was utilized to load, convert to mono, resample (to 16kHz), normalize, trim silence, and fix the duration of each audio sample to 3 seconds.

3.  **Spectrogram Generation:**
    *   **STFT (Short-Time Fourier Transform):** The `librosa.stft` function was used to compute the STFT for each processed audio signal. The amplitude spectrogram was then converted to decibels using `librosa.amplitude_to_db`.
    *   **Mel Spectrogram:** `librosa.feature.melspectrogram` was used to generate Mel-scaled spectrograms. Consistent with previous tasks, an amplitude-based Mel spectrogram (`power=1.0`) was computed and then converted to decibels.

4.  **Visualization:** A custom Python script, `src/instrument_comparison.py`, was developed to orchestrate the process. It generated a single Matplotlib figure with 11 rows (one for each instrument) and 2 columns (one for STFT, one for Mel-spectrogram). Each subplot displayed the respective spectrogram, allowing for direct visual comparison across instruments and between STFT and Mel representations for a given instrument.

5.  **Output:** The generated comparison plot was saved as `outputs/instrument_comparison.png` at 150 DPI.

## Tools Used
*   `librosa`: For audio loading, processing, and spectrogram computation.
*   `matplotlib`: For generating and saving the comparison plot.
*   `numpy`: For numerical operations.
*   `os`: For path manipulation.
*   `src/audio_processor.py`: Custom module for audio preprocessing.

## Results
The `instrument_comparison.png` image provides a comprehensive visual overview of how different instruments manifest in STFT and Mel-spectrogram representations. This visualization is crucial for understanding the unique timbral and harmonic characteristics that a CNN model would learn to differentiate. It highlights the distinct patterns and energy distributions across frequency and time for each instrument.

## Next Steps
Proceed to Task 6: CNN Model Development, leveraging the insights gained from this visual comparison for feature engineering and model architecture design.
