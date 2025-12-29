# InstruNet AI: CNN-Based Music Instrument Recognition

**InstruNet AI** is a deep learning project designed to automatically identify musical instruments in audio tracks using Convolutional Neural Networks (CNNs).

## Project Status

- **Phase:** Active Development
- **Current Focus:** Model Evaluation & Optimization (Task 10)
- **Dataset:** [IRMAS (Instrument Recognition in Musical Audio Signals)](https://www.upf.edu/web/mtg/irmas)

## Recent Progress
*   **Task 10 (In Progress):** Fixed critical Data Leakage; refactored pipeline to "Split-First, Augment-Second". Integrated `ReduceLROnPlateau` for training stability.
*   **Task 9 (Completed):** Performed initial robust training on 5x augmented dataset. Identified 64% real accuracy vs 89% leaked accuracy.
*   **Task 8 (Completed):** Initial CNN training and evaluation.
*   **Task 7 (Completed):** Designed and implemented the CNN architecture (`ModelTrainer`).

## Structure

- `src/`: Source code for audio processing, modeling, and evaluation.
- `datasets/`: Raw and processed IRMAS data.
- `docs/`: Technical reports for each development task.
- `outputs/`: Trained models, metrics, and visualization plots.

## Setup & Usage

1. **Environment:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Run Dataset Preprocessing (Updated):**
    Processes the IRMAS dataset using a **Split-First** strategy (80/20) to prevent data leakage. Augmentation is applied only to the training set.

    ```bash
    python3 src/preprocessing.py
    ```

    - **Train Data:** `datasets/IRMAS-ProcessedTrainingData/train/spectrograms/`
    - **Val Data:** `datasets/IRMAS-ProcessedTrainingData/validation/spectrograms/`

3. **Run Model Training:**
    Trains the CNN using the processed train/val splits.

    ```bash
    python3 src/training.py
    ```

4. **Run Instrument Comparison Visualization:**
    To generate a visual comparison of STFT and Mel-spectrograms for different instruments:

    ```bash
    python3 src/instrument_comparison.py
    ```

    - Comparison plot: `outputs/instrument_comparison.png`

