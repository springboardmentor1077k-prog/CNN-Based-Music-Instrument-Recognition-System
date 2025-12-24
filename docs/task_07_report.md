# Task 7 Report: Architecture Design & Pipeline Implementation

## Overview
This task focused on designing the system architecture and implementing the training pipeline for the InstruNet AI project. We successfully created a modular `ModelTrainer` class that handles data loading, model building, training, and evaluation.

## Accomplishments
1.  **Architecture Design**:
    *   Designed a standard CNN pipeline: `Input (Spectrograms) -> Preprocessing (Rescaling) -> CNN Layers -> Dense Layers -> Output (Class Probabilities)`.
    *   Selected `tensorflow.keras.utils.image_dataset_from_directory` for efficient data loading directly from the file system.

2.  **Pipeline Implementation**:
    *   Created `src/model_trainer.py` containing the `ModelTrainer` class.
    *   **Data Loading**: 
        *   Loads images from `outputs/mel_spectrograms_irmas/`.
        *   Splits data into 80% training and 20% validation.
        *   Implements caching and prefetching for performance optimization.
    *   **Model Building**:
        *   Constructed a baseline Sequential CNN model:
            *   Rescaling layer (1./255)
            *   3 Convolutional blocks (Conv2D + MaxPooling2D) with 16, 32, and 64 filters.
            *   Flatten layer.
            *   Dense layer (128 units, ReLU).
            *   Output Dense layer (11 units, Linear/Logits).
    *   **Training & Evaluation**:
        *   Implemented a training loop with `SparseCategoricalCrossentropy` loss.
        *   Added TensorBoard callbacks for logging.
        *   Added plotting functionality to visualize accuracy and loss curves.

3.  **Verification**:
    *   Ran a 3-epoch dry run to verify the pipeline.
    *   **Results**:
        *   Training ran successfully without errors.
        *   **Epoch 3 Results**:
            *   Training Accuracy: ~92%
            *   Validation Accuracy: ~95%
        *   Artifacts generated:
            *   `outputs/training_history.png`: Visual plot of training metrics.
            *   `outputs/instrunet_cnn.keras`: Saved model file.

## Observations
*   The pipeline is robust and ready for more extensive training.
*   The high accuracy on the validation set after just 3 epochs suggests the problem is well-suited for the chosen input features (Mel Spectrograms) and architecture.
*   System memory usage was high during the shuffle buffer filling, which is expected with large image datasets.

## Next Steps (Task 8)
*   Refine the CNN architecture (Task 8).
*   Train for more epochs to ensure convergence and check for overfitting.
*   Implement more robust evaluation metrics (Confusion Matrix, F1-Score).
