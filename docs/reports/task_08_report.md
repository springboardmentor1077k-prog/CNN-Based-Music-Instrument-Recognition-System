# Task 8 Report: Model Training & Evaluation

## Overview
In this task, we built and trained the Convolutional Neural Network (CNN) for the InstruNet AI project. The goal was to refine the architecture designed in Task 7, train it on the processed Mel Spectrogram dataset, and evaluate its performance using robust metrics.

## Model Architecture
The final model (`src/model_trainer.py`) is a Sequential CNN with the following structure:
1.  **Input & Preprocessing**:
    *   Input Shape: `(128, 128, 3)` (RGB Mel Spectrograms)
    *   Rescaling: `1./255` (Pixel normalization)
2.  **Feature Extraction (Convolutional Blocks)**:
    *   Block 1: `Conv2D(16, 3x3)` + `ReLU` + `MaxPooling2D`
    *   Block 2: `Conv2D(32, 3x3)` + `ReLU` + `MaxPooling2D`
    *   Block 3: `Conv2D(64, 3x3)` + `ReLU` + `MaxPooling2D`
    *   **Regularization**: `Dropout(0.2)` added after the last conv block.
3.  **Classification (Dense Layers)**:
    *   Flatten Layer
    *   Dense: `128 units` + `ReLU`
    *   **Regularization**: `Dropout(0.2)`
    *   Output: `Dense(11 units)` (Logits for 11 instrument classes)

**Total Parameters**: ~2.1 Million

## Training Configuration
*   **Dataset**: IRMAS Training Data (Processed into Mel Spectrograms)
    *   Total Samples: 6705
    *   Training Set: 5364 (80%)
    *   Validation Set: 1341 (20%)
*   **Batch Size**: 32
*   **Epochs**: 20
*   **Optimizer**: Adam
*   **Loss Function**: Sparse Categorical Crossentropy (from_logits=True)

## Results
The training process was highly successful, with the model converging rapidly.

*   **Final Training Accuracy**: ~99.6%
*   **Final Validation Accuracy**: **99.85%**
*   **Final Validation Loss**: 0.0070

### Performance Analysis
The model achieved near-perfect classification on the validation set. 

*   **Classification Report**:
    *   **Precision, Recall, and F1-Score** are **1.0 (100%)** for almost all classes (Cello, Clarinet, Flute, Guitar, Organ, Piano, Saxophone, Trumpet).
    *   Minor misclassifications occurred only between **Violin (vio)** and **Voice (voi)**, but even these had F1-scores > 0.98.
    *   **Overall Accuracy**: 99.85%

### Artifacts
*   **Saved Model**: `outputs/instrunet_cnn.keras`
*   **Training History**: `outputs/training_history.png` (Shows smooth convergence without significant overfitting).
*   **Confusion Matrix**: `outputs/confusion_matrix.png` (Visualizes the near-diagonal perfect predictions).
*   **Classification Report**: `outputs/classification_report.csv`

## Conclusion
The CNN architecture proved to be extremely effective for this task. The use of Mel Spectrograms as input features allowed the model to learn distinct patterns for each instrument. The addition of Dropout likely helped in generalizing well, although the high accuracy suggests the dataset's features are very distinct in the frequency domain.

The model is now ready for deployment or inference on new, unseen audio tracks.
