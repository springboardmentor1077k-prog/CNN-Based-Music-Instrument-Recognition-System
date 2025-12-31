Instrunet AI
Multilabel Musical Instrument Recognition Using CNN
Training and Prediction-Only Testing on IRMAS Dataset
1. Introduction

This project focuses on multilabel musical instrument recognition using a Convolutional Neural Network (CNN).
Unlike single-label classification, multilabel classification allows multiple instruments to be present in a single audio clip.

The goal of this work is to:

Train a CNN-based model for multilabel instrument detection

Perform prediction-only testing on the IRMAS testing dataset

Analyze model behavior through predicted probabilities and outputs

2. Dataset Description
2.1 Training Dataset

Source: IRMAS-based multilabel audio data

Type: Multilabel (multiple instruments per audio)

Preprocessing:

Audio resampled to 16 kHz

Silence trimming

Fixed duration padding/truncation (5 seconds)

Conversion to Mel-spectrogram images

2.2 Testing Dataset

Source: IRMAS Testing Dataset (downloaded via KaggleHub)

Type: Audio-only dataset (no multilabel ground truth available)

Usage:

Used only for prediction-only testing

No quantitative metrics (precision/recall/F1) computed

3. Audio Preprocessing Pipeline

Each test audio file undergoes the following steps:

Load audio and resample to 16 kHz

Normalize amplitude

Remove silent regions

Pad or truncate to 5 seconds

Convert to Mel-spectrogram

Convert Mel-spectrogram to image format (128×128)

This preprocessing ensures consistency between training and testing inputs.

4. Model Architecture

Model Type: Convolutional Neural Network (CNN)

Input: Mel-spectrogram images

Output: Probability scores for each instrument class

Activation: Sigmoid (for multilabel classification)

Instrument Classes
cello, clarinet, flute, guitar, organ,
piano, saxophone, trumpet, violin, voice, others

5. Training Procedure

Model trained using multilabel binary cross-entropy loss

Threshold-based decision applied during prediction

Saved trained model used for testing without retraining

6. Multilabel Prediction-Only Testing
6.1 Testing Strategy

All .wav files from the IRMAS testing dataset are processed

Each audio file is passed through the trained CNN

Output probabilities are thresholded at 0.2

Instruments with scores ≥ threshold are considered present

6.2 Example Prediction Output
File: brian eno - apollo- atmospheres and soundtracks - 01 - under stars-2.wav
Predicted instruments: flute, trumpet, violin
Raw scores: [0.028, 0.178, 0.466, 0.045, 0.097, 0.049, 0.041, 0.363, 0.247, 0.150, 0.072]


This result shows the model successfully identifying multiple instruments in a single audio clip.

7. Results Storage

All predictions are saved to a CSV file:

instrunet_multilabel_test_predictions.csv


The file contains:

Audio filename

Predicted instrument labels

Raw probability scores for each class

8. Observations

The model produces meaningful multilabel predictions

Different instruments receive varying confidence scores

Threshold selection plays a critical role in output quality

Prediction-only testing is appropriate due to lack of multilabel ground truth

9. Limitations

IRMAS testing dataset does not provide multilabel annotations

Quantitative evaluation metrics (precision, recall, F1) cannot be computed

Results are analyzed qualitatively based on prediction behavior

10. Conclusion

This project demonstrates a complete pipeline for multilabel musical instrument recognition, including:

CNN training on Mel-spectrogram images

Robust audio preprocessing

Prediction-only testing on real-world audio data

The approach validates the feasibility of multilabel instrument detection using CNNs and provides a strong foundation for future evaluation on fully annotated datasets.