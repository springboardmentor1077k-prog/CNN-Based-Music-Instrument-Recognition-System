# Final Model Submission: Instrunet AI - CNN Architecture

**Date:** January 21, 2026
**Model Version:** Iteration 11 (Refined Custom CNN + SpecAugment)
**Status:** Final Model for Evaluation

## 1. Model Overview
The **Instrunet AI** model is a specialized Convolutional Neural Network (CNN) designed for the classification of musical instruments from raw audio signals. It operates by converting audio into visual representations (Mel-Spectrograms) and treating the task as an image classification problem. The model has been optimized for the IRMAS dataset, achieving an overall test accuracy of **82.2%**.

## 2. Architecture Specification
The architecture is a deep, custom-designed CNN inspired by VGG/ResNet principles, specifically tuned for spectrogram texture analysis.

-   **Input Layer:** Accepts **224x224x3** RGB Mel-Spectrograms.
-   **Convolutional Blocks (Feature Extraction):**
    1.  **Block 1:** Conv2D (32 filters, 5x5 kernel) + BatchNormalization + MaxPooling2D. *Large kernel captures broad spectral shapes.*
    2.  **Block 2:** Conv2D (64 filters, 3x3 kernel) + BatchNormalization + MaxPooling2D.
    3.  **Block 3:** Conv2D (128 filters, 3x3 kernel) + BatchNormalization + MaxPooling2D.
    4.  **Block 4:** Conv2D (256 filters, 3x3 kernel) + BatchNormalization + MaxPooling2D. *Deep features for timbre differentiation.*
-   **Regularization Stack:**
    -   **Spatial Dropout (0.5):** Applied before the dense layers to prevent feature co-adaptation.
    -   **L2 Regularization (0.001):** Applied to the dense kernel weights.
-   **Classification Head:**
    -   **Global Average Pooling:** Reduces spatial dimensions to a single vector, making the model robust to temporal shifts.
    -   **Dense Layer:** 256 units + ReLU activation.
    -   **Output:** 11 units + **Softmax** activation (Single-Label Classification).

## 3. Training Methodology
-   **Data Augmentation (SpecAugment):** Implemented on-the-fly **Time Masking** and **Frequency Masking**. This creates vertical and horizontal "cuts" in the spectrogram, forcing the model to classify instruments based on partial information (context) rather than memorizing specific frequency bands.
-   **Loss Function:** Categorical Crossentropy.
-   **Optimizer:** Adam with dynamic Learning Rate Reduction (`ReduceLROnPlateau`).
-   **Input Resolution:** Standardized to 224x224 pixels to ensure sufficient frequency resolution for distinguishing harmonically similar instruments (e.g., Cello vs. Violin).

## 4. Performance Metrics
The model was evaluated on the IRMAS testing set using a windowed-voting strategy (aggregating predictions across 3-second segments of each track).

### Overall Metrics
-   **Test Accuracy:** **82.24%**
-   **Macro F1-Score:** **0.81**
-   **Weighted F1-Score:** **0.82**

### Per-Class Analysis
The model demonstrates exceptional performance on spectrally distinct instruments but faces expected challenges with overlapping timbres in the string and wind families.

| Instrument | Precision | Recall | F1-Score | Performance Tier |
| :--- | :--- | :--- | :--- | :--- |
| **Voice (voi)** | 0.87 | **0.98** | **0.92** | ⭐ **Tier 1 (Excellent)** |
| **Organ (org)** | 0.91 | 0.93 | **0.92** | ⭐ **Tier 1 (Excellent)** |
| **Acoustic Guitar** | 0.88 | 0.88 | 0.88 | Tier 1 (Excellent) |
| **Trumpet (tru)** | 0.84 | 0.91 | 0.87 | Tier 1 (Excellent) |
| **Piano (pia)** | 0.91 | 0.83 | 0.87 | Tier 1 (Excellent) |
| **Electric Guitar** | 0.83 | 0.75 | 0.79 | Tier 2 (Good) |
| **Clarinet (cla)** | 0.78 | 0.79 | 0.79 | Tier 2 (Good) |
| **Cello (cel)** | 0.68 | 0.87 | 0.76 | Tier 2 (Good) |
| **Flute (flu)** | 0.68 | 0.75 | 0.71 | Tier 3 (Average) |
| **Saxophone (sax)** | 0.74 | 0.68 | 0.71 | Tier 3 (Average) |
| **Violin (vio)** | 0.80 | 0.64 | 0.71 | Tier 3 (Average) |

## 5. Future Model Improvements
To further push the accuracy beyond 85% and address the "Tier 3" instrument confusion, the following model-centric improvements are proposed:

1.  **Multi-Scale Feature Fusion:** Implement a parallel branch architecture (Inception-style) where different kernel sizes (3x3, 5x5, 7x7) process the input simultaneously. This would allow the model to capture both transient attacks (short time) and sustained harmonics (long time) more effectively.
2.  **Attention Mechanisms:** Integrate **Squeeze-and-Excitation (SE) Blocks** to dynamically re-weight channel importance. This helps the model focus on "instrument-specific" channels while suppressing background noise channels.
3.  **Polyphonic Training (Multi-Label):** Transition the loss function from `Softmax` (Single-Label) to `Sigmoid` (Multi-Label). Training on polyphonic datasets (like NSynth combinations) would allow the model to explicitly learn to disentangle overlapping frequencies, likely improving performance on the complex Violin/Cello confusion cases.