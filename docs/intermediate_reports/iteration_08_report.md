# Iteration 8: Spatial Regularization

**Date:** January 03, 2026
**Baseline:** Iteration 7 (Label Smoothing, 70.23% Test Acc)
**Status:** In Progress (Training Pending)

## 1. Objective
Close the 11% gap between Validation (81%) and Test (70%) accuracy by implementing stronger regularization tailored for spectrogram data.

## 2. Changes Applied (Architecture)
*   **SpatialDropout2D:** Replaced standard Dropout in the convolutional blocks with progressive SpatialDropout2D (0.1 in early layers, 0.2 in deep layers).
    *   *Rationale:* Standard Dropout is weak for correlated spectrogram pixels. SpatialDropout2D drops entire feature maps, forcing the model to learn redundant and more robust feature representations (e.g., recognizing an instrument by both its attack and its harmonic series, rather than just one).

## 3. Changes Applied (Inference)
*   **Aggregation Reversion:** Reverted `src/testing.py` to **Global Mean Pooling** after Iteration 7 showed that Top-K Pooling was counter-productive for this dataset.

## 4. Next Steps
1.  Run the Iteration 8 training run (50 epochs) on the HPC.
2.  Evaluate if SpatialDropout2D successfully narrows the generalization gap.
