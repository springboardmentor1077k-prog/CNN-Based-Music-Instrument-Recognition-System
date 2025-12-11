# InstruNet AI: CNN-Based Music Instrument Recognition

**InstruNet AI** is a deep learning project designed to automatically identify musical instruments in audio tracks using Convolutional Neural Networks (CNNs).

## Project Status

- **Phase:** Active Development
- **Current Focus:** Model Training Preparation
- **Dataset:** [IRMAS (Instrument Recognition in Musical Audio Signals)](https://www.upf.edu/web/mtg/irmas)

## Recent Progress
*   **Task 6 (Completed):** Implemented audio augmentation (Noise, Time Stretch, Pitch Shift) and visualization.
*   **Task 5 (Completed):** Created project demo notebook.
*   **Task 4 (Completed):** Finalized preprocessing pipeline.

## Structure

- `src/`: Source code for audio processing and modeling.
- `datasets/`: Contains the IRMAS training data.
- `docs/`: Documentation and task reports.
- `outputs/`: Generated plots and logs.
- `tasks/`: Task tracking.

## Setup & Usage

1. **Environment:**
    Activate the Conda environment:

    ```bash
    conda activate instrunet_env
    ```

    Then install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. **Run Audio Augmentation Demo (New):**
    To generate augmented audio samples and comparison spectrograms:

    ```bash
    python3 src/demo_augmentation.py
    ```
    
    - Results: `outputs/augmented_samples/`

3. **Run Dataset Preprocessing:**
    To process the entire IRMAS dataset (Normalize, Trim, Pad to 3s) and generate Mel Spectrograms:

    ```bash
    python3 src/process_dataset.py
    ```

    - Processed Audio (WAV): `outputs/processed_irmas/`
    - Mel Spectrograms (PNG): `outputs/mel_spectrograms_irmas/`

4. **Run Instrument Comparison Visualization:**
    To generate a visual comparison of STFT and Mel-spectrograms for different instruments:

    ```bash
    python3 src/instrument_comparison.py
    ```

    - Comparison plot: `outputs/instrument_comparison.png`

