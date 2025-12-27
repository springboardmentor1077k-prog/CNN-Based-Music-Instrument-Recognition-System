# Task 09 Report: Model Optimization (Batch Normalization)

**Date:** December 27, 2025  
**Status:** In Progress (Architecture Optimized)

## 1. Overview
The goal of Task 09 is to optimize the CNN model to handle the increased variance and volume of the augmented dataset (33,000+ samples). This report documents the architectural improvements made to improve training stability and convergence speed.

## 2. Implementation: Batch Normalization
To improve the model's robustness against "Internal Covariate Shift"—especially with noisy and pitch-shifted data—I have integrated **Batch Normalization** into the CNN architecture.

### 2.1. Architectural Changes
The model in `src/model_trainer.py` was updated to include Batch Normalization layers after every convolutional block and the primary dense layer.

**Updated Layer Stack:**
1.  **Input:** (128, 128, 3) Mel Spectrogram.
2.  **Rescaling:** Normalize pixels to [0, 1].
3.  **Conv Block 1:** 16 filters -> **BatchNormalization** -> MaxPool.
4.  **Conv Block 2:** 32 filters -> **BatchNormalization** -> MaxPool.
5.  **Conv Block 3:** 64 filters -> **BatchNormalization** -> MaxPool.
6.  **Regularization:** Dropout (0.2).
7.  **Classification:**
    *   Flatten.
    *   Dense (128) -> **BatchNormalization** -> Dropout (0.2).
    *   Output (11 classes, Logits).

### 2.2. Benefits for InstruNet AI
*   **Faster Convergence:** Normalizing activations allows for potentially higher learning rates and fewer epochs to reach high accuracy.
*   **Regularization Effect:** Batch Normalization provides a slight regularization effect, which complements our existing Dropout layers.
*   **Handling Augmentation:** Since we now have 5 variants per audio file (Original, Noise, Pitch, Vol, Shift), Batch Norm ensures the model can learn consistent features across these diverse inputs.

## 3. Next Steps
*   Execute the training pipeline on Google Colab using the **T4 GPU**.
*   Monitor validation loss to ensure the gap between training and validation accuracy remains small (preventing overfitting).
*   Proceed to Task 10 for advanced metric evaluation (ROC/AUC).
