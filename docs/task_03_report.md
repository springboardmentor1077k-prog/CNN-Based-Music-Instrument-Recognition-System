# Task 3 Report: NSynth Dataset Exploration and Data Pipeline Setup

## Objective
The primary objective of Task 3 was to explore the NSynth dataset, inspect its metadata, load sample audio files, extract labels, and create an instrument mapping table. This task was performed as a separate exploration, acknowledging a potential future transition from the IRMAS dataset to NSynth for the main project.

## Steps Performed

1.  **Discrepancy Resolution:** Identified and clarified the discrepancy between project documentation (GEMINI.md, README.md pointing to IRMAS) and `tasks/task_03.md` (requesting NSynth). The user confirmed to proceed with NSynth as a separate task.
2.  **`GEMINI.md` Update:** Updated `GEMINI.md` to reflect the focus of Task 3 on NSynth and the potential future dataset transition.
3.  **NSynth Dataset Discovery:** Discovered that the `nsynth-train.jsonwav.tar.gz` archive was already present in the `datasets/` directory, eliminating the need for a large download.
4.  **Dataset Extraction:** The `nsynth-train.jsonwav.tar.gz` archive was extracted into `datasets/nsynth-training-data/`. The contents were found within a nested directory: `datasets/nsynth-training-data/nsynth-train/`, containing `audio/` and `examples.json`.
5.  **Metadata Inspection:** Inspected the `datasets/nsynth-training-data/nsynth-train/examples.json` file. Key metadata entries observed include:
    *   `note_str`: Unique identifier (e.g., "guitar_acoustic_001-082-050")
    *   `instrument_str`: Specific instrument (e.g., "guitar_acoustic_001")
    *   `instrument_family_str`: Broader instrument family (e.g., "guitar")
    *   `pitch`, `velocity`, `qualities_str`.
    This satisfied the requirement to "Inspect at least 3 metadata entries".
6.  **Data Processing Script Development:** Created `src/nsynth_data_processor.py` to:
    *   Load `examples.json`.
    *   Parse metadata to extract instrument information and construct audio file paths.
    *   Create a Pandas DataFrame as the instrument mapping table.
    *   Load and print information for 3 sample audio files using `librosa`, confirming successful audio loading and label extraction.
7.  **Dependency Management:** Added `pandas` to `requirements.txt` and installed it to ensure the script runs correctly.
8.  **Script Execution:** Executed `src/nsynth_data_processor.py` after activating the virtual environment, which successfully generated the mapping table and loaded sample audio.

## Results

*   **Metadata:** The NSynth dataset's `examples.json` provides rich metadata including specific instrument names, instrument families, pitch, velocity, and descriptive qualities for each audio sample.
*   **Sample Audio Loading:** The `src/nsynth_data_processor.py` successfully loaded 3 sample WAV files (e.g., `guitar_acoustic_001-082-050.wav`, `bass_synthetic_120-108-050.wav`, `organ_electronic_120-050-127.wav`) with a sample rate of 16000 Hz and a duration of 4 seconds, confirming the ability to access and process the audio data.
*   **Instrument Mapping Table:** A CSV file named `outputs/nsynth_mapping_table.csv` was generated. This file contains 289,205 entries, each detailing the `file_name`, `instrument`, `family`, `source` (NSynth), and `full_path` for every audio sample in the NSynth training set. This table serves as the primary data structure for organizing and accessing the dataset for future model training.

## Conclusion

Task 3 has successfully established a preliminary data pipeline for the NSynth dataset. The metadata has been inspected, sample audio files can be loaded, and a comprehensive mapping table has been created. This lays the groundwork for integrating the NSynth dataset into the InstruNet AI project for CNN training, especially if the project formally transitions away from the IRMAS dataset.

## Artifacts

*   `src/nsynth_data_processor.py`: Python script for processing NSynth data.
*   `outputs/nsynth_mapping_table.csv`: CSV file containing the instrument mapping table for the NSynth training dataset.
*   `requirements.txt`: Updated to include `pandas`.
*   `tasks/task_03.md`: Updated to mark all sub-tasks as complete.
*   `GEMINI.md`: Updated to reflect the status of Task 3 and future considerations.
