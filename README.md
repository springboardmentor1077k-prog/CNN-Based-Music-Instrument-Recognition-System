# CNN-Based-Music-Instrument-Recognition-System

In this project, the goal is to build a deep learning system that can identify which musical instruments are present in an audio recording.
So far, the following steps have been completed:

🔹 1. Dataset Loading

The FMA (Free Music Archive) dataset was loaded directly from the Git repository.

Audio files were accessed in .mp3 format and stored for further processing.

🔹 2. Audio Exploration

A sample audio file was loaded and inspected.

The waveform of the audio was generated to visualize amplitude variation over time for both left and right stereo channels.

🔹 3. Time–Frequency Feature Extraction

Three main audio representations were generated:
| Representation                                 | Purpose                                                            |
| ---------------------------------------------- | ------------------------------------------------------------------ |
| **STFT Spectrogram (Normal Spectrogram)**      | Shows how raw frequencies change over time                         |
| **Mel Spectrogram**                            | Time–frequency features mapped to human auditory scale             |
| **MFCC (Mel Frequency Cepstral Coefficients)** | Compact version of mel features, mostly used in speech recognition |

🎯 Why We Use Mel Spectrograms for CNN-Based Instrument Recognition

Musical instruments are primarily distinguished by their timbre — their unique harmonic texture and tone color.
Mel Spectrogram is best for the Projects like "CNN-Based Music Instrument Recognition System" as Musical instruments are recognized by their timbre, which depends on harmonic structures that are more easily captured on the Mel scale, where low frequencies (where most musical harmonics lie) are emphasized and high frequencies are compressed according to human hearing.

MFCC is a compressed version of the Mel spectrogram that keeps only a small number of coefficients. This compression is helpful for speech recognition but removes high-frequency harmonic details that are crucial for identifying musical instruments (e.g., violin bow friction, guitar string resonances, cymbal overtones). Therefore, although MFCC keeps the “most important” timbral features for speech, it throws away too much harmonic information for musical instruments, making it less suitable for CNNs.
