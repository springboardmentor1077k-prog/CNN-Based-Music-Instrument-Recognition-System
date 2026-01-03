Report on Multilabel Instrument Classification Model
1. Introduction

This report describes the development and evaluation of my first multilabel deep learning model for musical instrument classification using the IRMAS dataset. Unlike single-label classification, multilabel learning allows the model to predict multiple instruments simultaneously from the same audio sample, which better reflects real-world music recordings where several instruments often co-occur.

The primary objective of this work was to design, train, and evaluate a Convolutional Neural Network (CNN) capable of learning multilabel instrument representations from audio features, and to understand the challenges involved in multilabel audio classification.

2. Dataset Description

The IRMAS (Instrument Recognition in Musical Audio Signals) dataset was used for this experiment. It contains audio recordings annotated with instrument labels. For multilabel training, audio samples may contain more than one instrument label, making the task more complex than traditional single-label classification.

Key Characteristics:

Audio-based dataset focused on instrument recognition

Multiple instruments can be present in a single audio clip

Suitable for evaluating multilabel classification performance

3. Feature Extraction

To make the audio data suitable for CNN-based learning, each audio file was converted into a Mel-spectrogram representation.

Rationale:

Mel-spectrograms capture both time and frequency information

They align well with human auditory perception

CNNs are effective at learning spatial patterns from spectrogram images

The extracted Mel-spectrograms were used as input to the neural network.

4. Model Architecture

A Convolutional Neural Network (CNN) was designed for multilabel classification.

Architecture Overview:

Multiple Conv2D layers for feature extraction

MaxPooling layers to reduce spatial dimensions

Flatten layer to convert feature maps into vectors

Fully connected (Dense) layers for classification

Dropout layers to reduce overfitting

Sigmoid activation in the output layer to enable multilabel predictions

Why Sigmoid?

Unlike softmax (used for single-label tasks), sigmoid allows each instrument class to be predicted independently, which is essential for multilabel problems.

5. Training Strategy

Loss Function: Binary Cross-Entropy (suitable for multilabel classification)

Optimizer: Adam optimizer for efficient gradient-based learning

Metrics: Accuracy, Precision, and Recall

Data Split: Training and validation sets were used to monitor learning performance

Regularization techniques such as Dropout were applied to improve generalization.

6. Evaluation and Results

The model showed the ability to:

Learn meaningful representations of musical instruments

Predict multiple instruments for a single audio input

Generalize reasonably well without severe overfitting

Observations:

Training and validation trends indicate stable learning behavior

Precision and Recall metrics highlight the trade-off between false positives and false negatives

Some instruments are easier to detect than others due to timbral distinctiveness and dataset balance

As this is a first multilabel model, the results are encouraging and demonstrate a successful transition from single-label to multilabel learning.

7. Challenges Faced

Multilabel learning is inherently more complex than single-label classification

Instrument overlap in audio increases ambiguity

Class imbalance affects prediction performance

Threshold selection for sigmoid outputs impacts final predictions

8. Conclusion

This work successfully demonstrates the implementation of a first multilabel CNN model for instrument recognition using the IRMAS dataset. The model is capable of identifying multiple instruments from a single audio sample and provides a strong foundation for further research.

Overall, this experiment helped in:

Understanding multilabel classification concepts

Gaining hands-on experience with audio-based CNN models

Learning appropriate loss functions and evaluation metrics for multilabel tasks