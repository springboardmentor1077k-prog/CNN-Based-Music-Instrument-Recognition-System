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


# Working With Nsynth Dataset:

In this phase of the project, we shifted from the FMA dataset to the NSynth dataset, which is specifically designed for musical instrument recognition.
Unlike FMA, NSynth contains isolated instrument notes along with detailed metadata and labels, making it highly suitable for building a CNN-based instrument classification model.

1. Dataset Loading

The NSynth-mini dataset was loaded directly into the notebook using Hugging Face.
The dataset provides both:

raw audio (stored as WAV bytes)

structured metadata (pitch, instrument family, velocity, qualities, etc.)

The dataset was successfully loaded into a pandas DataFrame for further processing.

2. Metadata Inspection

Multiple metadata entries were inspected to understand the structure of the dataset. This step confirmed that the dataset contains rich musical information and is perfectly structured for supervised learning.

3. Audio Extraction & Playback

Since audio files are stored as raw WAV byte streams, they were decoded from the JSON-like structure and converted into waveform arrays using soundfile.

The decoded audio was:

played using an audio player in the notebook

visualized as a waveform plot to analyze amplitude-time signatures

This confirmed the audio integrity and usability for spectrogram generation later.

4. Instrument Label Extraction

Instrument labels were extracted in two forms:

✔ Human-readable labels

(e.g., mallet, flute, bass, string, …)

✔ Numeric class IDs

(e.g., 0, 1, 2, 3 …)

Both types were compiled to understand class distribution and for later model training.

5. Instrument Mapping Table

A mapping table was formed to link numeric class IDs with instrument family names.

### Status Summary

So far on the NSynth dataset, the following tasks have been successfully completed:

Dataset loaded using Hugging Face

Inspected multiple metadata entries

Extracted and decoded audio signals from raw bytes

Played and visualized the audio waveforms

Printed instrument labels (from JSON + dataframe)

Created numeric-to-string instrument mapping table
