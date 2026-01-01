# Task 10 Report: Detailed Model Evaluation

**Date:** January 01, 2026  
**Model Version:** High-Resolution (224x224) + Class Weighting  
**Status:** Completed

## 1. Overview
The goal of Task 10 was to perform a deep-dive evaluation of the InstruNet AI model following the infrastructure hardening and resolution optimization. We moved beyond simple accuracy to analyze per-class precision/recall, confusion patterns, and ROC-AUC metrics.

## 2. Global Metrics
- **Overall Accuracy:** 73.48%
- **Macro Average F1-Score:** 0.72
- **Weighted Average F1-Score:** 0.73

## 3. Per-Class Performance
The model shows significant variance across different instrument types:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Voice (voi)** | 0.91 | 0.92 | 0.91 | 156 |
| **Piano (pia)** | 0.86 | 0.90 | 0.88 | 145 |
| **Organ (org)** | 0.83 | 0.88 | 0.86 | 137 |
| **Acoustic Guitar (gac)** | 0.78 | 0.84 | 0.80 | 128 |
| **Trumpet (tru)** | 0.72 | 0.70 | 0.71 | 116 |
| **Electric Guitar (gel)** | 0.66 | 0.70 | 0.68 | 152 |
| **Clarinet (cla)** | 0.64 | 0.70 | 0.67 | 101 |
| **Cello (cel)** | 0.57 | 0.71 | 0.63 | 78 |
| **Flute (flu)** | 0.68 | 0.59 | 0.63 | 91 |
| **Saxophone (sax)** | 0.68 | 0.53 | 0.60 | 126 |
| **Violin (vio)** | 0.58 | 0.45 | 0.50 | 116 |

## 4. Key Findings

### 4.1. Success Stories
- **Voice, Piano, and Organ** remain the strongest classes, all achieving F1-scores above 85%. Their spectral signatures are highly distinct and were well-captured by the CNN.
- **Class Weighting Impact:** The **Cello (cel)** recall improved significantly to **70.5%** (previously ~56%), proving that the model is now better at identifying this minority class.

### 4.2. Failure Modes
- **Violin (vio)** is the primary weak point (Recall: 45%). Confusion matrix analysis shows it is frequently confused with Cello and Saxophone.
- **Saxophone (sax)** recall is surprisingly low (53%), often being misclassified as Clarinet or Trumpet due to shared harmonic characteristics in the mid-range.

## 5. Visual Artifacts
The following plots were generated in the `outputs/` directory:
- `normalized_confusion_matrix.png`: Shows the specific "Instrument Swapping" patterns.
- `roc_curves.png`: Displays high AUC (>0.90) for most classes, indicating strong discriminative potential even if the hard thresholding (accuracy) is lower.
- `training_history.png`: Confirms smooth convergence with `EarlyStopping` triggered at epoch 27.

## 6. Optimization Strategies (Moving Forward)
Based on this analysis and the "Day 22" notes, the next steps are:
1.  **Error First Analysis:** Manually inspect the top 5 overconfident mistakes to determine if they are "Confusion" or "Hallucination" errors.
2.  **Filter Tuning:** Increase model capacity (Filters: 32 -> 64 -> 128 -> 256) to help distinguish between Cello and Violin.
3.  **Stress Testing:** Measure model robustness against volume and noise variations.

---
**Status:** Task 10 items fully addressed and saved.
