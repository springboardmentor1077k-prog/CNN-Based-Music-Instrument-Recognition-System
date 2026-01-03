# Iteration 7: Optimization & Regularization

**Date:** January 02, 2026
**Baseline:** Iteration 5 (SpecAugment, ~69% Test Acc)
**Objective:** Reduce Overfitting (99% Train vs 81% Val gap) & Improve Full-Song Evaluation

## 1. Current State (Iteration 6 - Label Smoothing)
We applied Label Smoothing (0.1) to the baseline.
-   **Validation Acc:** 81.35% (Improved from ~73%)
-   **Test Acc:** 70.23% (Improved from 69.09%)
-   **Status:** Overconfidence reduced, but significant generalization gap remains.

## 2. Hypothesis
The model is "memorizing" training artifacts due to high capacity and ineffective regularization (Standard Dropout). Additionally, the testing aggregation (Mean Pooling) punishes intermittent instruments in full songs.

## 3. Optimization Strategy (Iteration 7)

### A. Training Architecture
*   **SpatialDropout2D:** Replacing standard Dropout.
    *   *Why:* Spectrograms have high local correlation. Dropping pixels is weak; dropping entire *feature maps* forces the model to learn redundant, robust features (e.g., Timbre + Attack, not just one).

### B. Testing Inference
*   **Top-K Mean Pooling:** Replacing Global Mean Pooling.
    *   *Why:* IRMAS test files are full songs. An instrument might only appear for 15 seconds. Mean pooling dilutes this signal with 2 minutes of silence. Top-K (e.g., Top 33%) focuses on the most confident detections, ignoring the "noise" of the rest of the track.

## 4. Next Steps
1.  Modify `testing.py` to implement Top-K Pooling.
2.  Modify `model_trainer.py` to implement `SpatialDropout2D`.
3.  Retrain and Evaluate.
