# Spectrogram Comparison & Selection

## Overview
As part of **Task 2**, we generated three types of visual representations for the audio signal:
1.  **STFT (Short-Time Fourier Transform) Spectrogram**
2.  **Mel Spectrogram**
3.  **MFCC (Mel-Frequency Cepstral Coefficients)**

## Comparison

### 1. STFT Spectrogram
*   **Description:** Represents the magnitude of the frequency content of the signal over time. The frequency axis is linear.
*   **Pros:** Contains the most raw information about the signal.
*   **Cons:** High dimensionality. The linear frequency scale doesn't align well with human auditory perception (we hear logarithmically).

### 2. Mel Spectrogram
*   **Description:** A spectrogram where the frequencies are converted to the Mel scale, which mimics human ear perception (more resolution at lower frequencies, less at higher).
*   **Pros:** 
    *   Aligns with human perception.
    *   Reduces dimensionality compared to STFT while retaining more spectral texture than MFCCs.
    *   Standard input for modern CNN-based audio classifiers.
*   **Cons:** Lossy compared to raw STFT (but usually beneficial loss).

### 3. MFCCs
*   **Description:** Derived from the Mel Spectrogram by taking the Discrete Cosine Transform (DCT). It decorrelates the coefficients.
*   **Pros:** Very compact representation; standard for speech recognition (HMMs/GMMs).
*   **Cons:** discard some information (like pitch) that might be useful for instrument detection. The DCT step can remove spatial relationships that CNNs exploit.

## Conclusion & Selection
**Selected Representation: Mel Spectrogram**

**Reasoning:**
For Deep Learning models, especially **CNNs (Convolutional Neural Networks)**, the **Mel Spectrogram** is generally the preferred input. 
1.  **Structure:** It preserves the 2D "image-like" structure of the sound (Time vs. Frequency), allowing the CNN to learn patterns like harmonic stacks and formants, which are critical for distinguishing instruments.
2.  **Efficiency:** It filters out high-frequency detail that is less perceptually relevant, reducing the input size without losing critical timbre information.
3.  **Performance:** Literature on instrument recognition (e.g., the IRMAS dataset papers) typically shows superior performance with Mel Spectrograms compared to MFCCs for deep learning approaches.

We will proceed with **Mel Spectrograms** as the primary input feature for our CNN model.
