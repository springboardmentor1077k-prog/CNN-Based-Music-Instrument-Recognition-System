# Task 06 Report: Audio Augmentation

## Objective
Implement at least 3 audio augmentation techniques to increase dataset diversity and robustness of the model. Apply these techniques to a sample set of audios and visualize the results.

## Implemented Techniques
The following augmentation functions were added to `src/audio_processor.py`:

1.  **Noise Injection (`add_noise`)**: Adds Gaussian noise to the audio signal.
    *   *Parameters*: `noise_level` (default: 0.005)
2.  **Time Stretching (`time_stretch`)**: Changes the speed of the audio without affecting pitch.
    *   *Parameters*: `rate` (default: 1.0). >1.0 speeds up, <1.0 slows down.
3.  **Pitch Shifting (`pitch_shift`)**: Shifts the pitch of the audio by a specified number of semitones.
    *   *Parameters*: `n_steps` (default: 0).

## Execution & Verification
*   **Script**: `src/demo_augmentation.py` was created to demonstrate these techniques.
*   **Process**:
    1.  Selected 3 random WAV files from the IRMAS dataset.
    2.  Applied standard preprocessing (Mono, 16kHz, Normalize, Trim, Fix Duration).
    3.  Generated 3 variants for each file:
        *   Noise Added (level=0.02)
        *   Time Stretched (rate=1.2x)
        *   Pitch Shifted (+4 semitones)
*   **Outputs**:
    *   Generated files are saved in `outputs/augmented_samples/`.
    *   Filenames follow the pattern: `{original_name}_{augmentation_type}.wav`.

## Visualization Comparison
*   The `src/demo_augmentation.py` script generates a combined plot for each sample.
*   **Mel Spectrogram Comparison**:
    *   A single figure with 4 subplots (Original, Noise, Time Stretch, Pitch Shift).
    *   Allows visual inspection of how augmentations affect the frequency and time domains.
    *   Saved as `{original_name}_comparison.png` in `outputs/augmented_samples/`.

## Results
The augmentation pipeline is functional and integrates with the existing `audio_processor.py` module. These techniques can now be applied during the training phase (on-the-fly) or to generate an offline augmented dataset.

## Next Steps
*   Integrate these augmentations into the data loader or training loop.
*   Proceed to Model Training (Task 7).
