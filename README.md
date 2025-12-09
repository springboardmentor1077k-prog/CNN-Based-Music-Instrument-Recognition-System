# InstruNet AI: CNN-Based Music Instrument Recognition

**InstruNet AI** is a deep learning project designed to automatically identify musical instruments in audio tracks using Convolutional Neural Networks (CNNs).

## Project Status
- **Phase:** Active Development
- **Current Focus:** Preprocessing Pipeline Complete / CNN Model Preparation
- **Dataset:** [IRMAS (Instrument Recognition in Musical Audio Signals)](https://www.upf.edu/web/mtg/irmas)

## Structure
- `src/`: Source code for audio processing and modeling.
- `datasets/`: Contains the IRMAS training data.
- `docs/`: Documentation and task reports.
- `outputs/`: Generated plots and logs.
- `tasks/`: Task tracking.

## Setup & Usage

1.  **Environment:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Dataset Preprocessing:**
    To process the entire IRMAS dataset (Normalize, Trim, Pad to 3s) and generate Mel Spectrograms:
    ```bash
    python3 src/process_dataset.py
    ```
    - Processed Audio (WAV): `outputs/processed_irmas/`
    - Mel Spectrograms (PNG): `outputs/mel_spectrograms_irmas/`

3.  **Run Audio Processor (Single File):**
    To explore the dataset and generate a sample waveform:
    ```bash
    python3 src/audio_processor.py
    ```
    Artifacts will be saved to the `outputs/` directory.