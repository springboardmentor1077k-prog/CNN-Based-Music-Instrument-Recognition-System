# Task 02 Report: Spectrogram Generation & Visualization

**Date:** December 4, 2025  
**Status:** Completed

## 1. Overview

The objective of Task 02 was to establish the audio visualization pipeline. This involved converting raw audio waveforms into various spectral representations (Spectrograms) to identify the most suitable feature set for the upcoming CNN (Convolutional Neural Network) model.

## 2. Implementation Details

### 2.1. Code Structure

* **`src/visualizer.py`**: A new module created to handle all plotting logic. It supports:
  * **STFT Spectrograms:** Linear frequency scale.
  * **Mel Spectrograms:** Logarithmic frequency scale (mimicking human hearing). Includes support for both Power (`power_to_db`) and Amplitude (`amplitude_to_db`) scales.
  * **MFCCs:** Mel-Frequency Cepstral Coefficients.
* **`src/main.py`**: The orchestration script that:
    1. Loads audio using `src/audio_processor.py`.
    2. Generates a high-resolution Waveform plot (300 DPI).
    3. Calls `visualizer.py` to generate spectrograms.

### 2.2. Technical Decisions

* **Sample Rate:** maintained at **16kHz** (established in Task 1).
* **Resolution:** All output images are now saved at **300 DPI** for high quality.
* **Mel Spectrogram Calculation:** We explicitly switched to `librosa.power_to_db` for Mel Spectrograms.
  * *Reasoning:* `librosa.feature.melspectrogram` returns a power spectrogram (magnitude squared). Using `amplitude_to_db` on power data results in mathematically incorrect dB values (doubling them). The correction ensures accurate intensity representation.

## 3. Feature Selection

A comparison was conducted (see `docs/spectrogram_comparison.md`) to select the best input for the CNN.

* **Selected Feature:** **Mel Spectrogram**
* **Rationale:**
  * It preserves the time-frequency structure essential for CNNs (unlike MFCCs which can be too compressed).
  * It aligns with human auditory perception, focusing on frequencies relevant to instrument timbre.
  * It is the standard state-of-the-art input for audio classification tasks in deep learning literature.

## 4. Outputs

The following artifacts were generated in the `outputs/` directory using a sample Cello file from the IRMAS dataset:

1. `waveform_[filename].png`: Time-domain signal. *(Generation established in Task 1, saving integrated into pipeline in Task 2).*
2. `stft_spectrogram_[filename].png`: Raw frequency content.
3. `mel_spectrogram_power_[filename].png`: **(Primary Model Input)** Mel-scaled power spectrogram.
4. `mfcc_[filename].png`: Cepstral coefficients.

## 5. Conclusion

The visualization pipeline is robust and mathematically correct. We are ready to proceed to **Task 3**, which will involve building the data pipeline to batch-process the entire IRMAS dataset into these Mel Spectrograms and preparing the data for CNN training.
