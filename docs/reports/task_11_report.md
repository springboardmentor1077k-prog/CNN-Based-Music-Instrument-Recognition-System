# Task 11 Report: Hyperparameter Tuning Analysis

**Date:** January 02, 2026  
**Status:** Completed  
**Strategy:** Automated Grid Search (Parallel Sharding)

## 1. Objective
The goal of Task 11 was to systematically tune the model's "knobs" (Dropout, Learning Rate, and L2 Regularization) to break past the accuracy plateau and maximize instrument recognition metrics.

## 2. Tuning Grid
We executed 18 parallelized training runs (15 epochs each) on the HPC cluster:
- **Dropout:** [0.3, 0.4, 0.5]
- **Learning Rate:** [0.001, 0.0005, 0.0001]
- **L2 Regularization:** [0.001, 0.0001]

## 3. Top Performers

| Run ID | Dropout | LR | L2 | Val Accuracy | Val Loss |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **8 (Winner)** | **0.4** | **0.001** | **0.0001** | **82.84%** | **0.7298** |
| 1 | 0.3 | 0.001 | 0.001 | 0.8172 | 0.8647 |
| 14 | 0.5 | 0.001 | 0.0001 | 0.8165 | 0.7999 |
| 16 | 0.5 | 0.0005 | 0.0001 | 0.8165 | 0.8005 |

## 4. Key Deductions

### 4.1. Regularization Balance (Dropout & L2)
- **Finding:** Lowering Dropout to **0.4** and L2 to **0.0001** yielded the best results.
- **Deduction:** The previous baseline (DO=0.5, L2=0.001) was **over-regularizing** the model. Because we already use **SpecAugment**, the model was being penalized too heavily, which suppressed the learning of subtle instrument features. Relaxing these constraints allowed the model to "breathe" and capture finer details.

### 4.2. Learning Rate Stability
- **Finding:** **0.001** remains the optimal starting point for the Adam optimizer.
- **Deduction:** Lower rates (0.0001) were too slow to converge within the 15-epoch window, while 0.001 hit the 80% mark rapidly and remained stable.

### 4.3. High-Capacity Model Success
- The model consistently reached 80%+ across different configurations, proving that the **256-filter architecture** is robust and well-suited for the IRMAS dataset.

## 5. Optimized Configuration
For the final production training run (50 epochs), we will use:
- **Dropout:** 0.4
- **L2:** 0.0001
- **Learning Rate:** 0.001
- **Resolution:** 224x224
- **Strategy:** SpecAugment + Class Weighting

---
**Status:** Best hyperparameters identified. Ready for full training.
