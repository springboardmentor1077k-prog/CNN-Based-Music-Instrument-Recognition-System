# InstruNet AI: CNN-Based Music Instrument Recognition

**InstruNet AI** is a deep learning project designed to automatically identify musical instruments in audio tracks using Convolutional Neural Networks (CNNs).

## Project Status
- **Phase:** Active Development
- **Current Focus:** Audio Processing & Dataset Exploration
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

2.  **Run Audio Processor:**
    To explore the dataset and generate a sample waveform:
    ```bash
    python3 src/audio_processor.py
    ```
    Artifacts will be saved to the `outputs/` directory.