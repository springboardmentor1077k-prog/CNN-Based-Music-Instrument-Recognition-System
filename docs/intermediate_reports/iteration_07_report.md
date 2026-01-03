# Iteration 7: Label Smoothing & Aggregation Analysis

**Date:** January 03, 2026
**Baseline:** Iteration 5 (SpecAugment, ~69% Test Acc)
**Status:** Completed

## 1. Objective
Reduce model overconfidence and improve full-song test performance.

## 2. Changes Applied
1.  **Label Smoothing (0.1):** Replaced hard targets (0, 1) with soft targets (0.05, 0.95) using Categorical Crossentropy.
2.  **Top-K Pooling (33%):** Experimented with taking the mean of only the top 33% most confident windows per song during testing.

## 3. Results & Metrics

| Metric | Iteration 5 (Baseline) | Iteration 7 (Label Smoothing) |
| :--- | :--- | :--- |
| **Validation Acc** | ~73.5% | **81.35%** |
| **Test Acc (Mean)** | 69.09% | **70.23%** |
| **Test Acc (Top-K)** | N/A | 69.26% |

### Key Observations
-   **Success:** Label Smoothing effectively reduced the "Overconfidence Misconception." No more 100% confidence errors. Violin and Saxophone recall improved significantly.
-   **Failure:** Top-K Pooling (33%) lowered accuracy compared to Mean Pooling. This suggests that for IRMAS, the instrument signal is distributed enough that Mean Pooling remains the superior aggregation method.

## 4. Conclusion
Iteration 7 successfully broke the 70% Test Accuracy barrier. However, the ~11% gap between Validation and Test indicates that fundamental overfitting remains.