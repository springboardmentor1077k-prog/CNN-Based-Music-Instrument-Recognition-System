CNN Improvement – Music Instrument Classification
📌 Overview

This module presents an improved CNN architecture for Music Instrument Classification using Mel Spectrograms.
It builds upon the baseline CNN by applying best practices used in modern deep learning models.

Key improvements over the baseline CNN:

Batch Normalization after every convolution and dense layer

Dropout for regularization and overfitting control

Increased network depth with an additional convolution block

Increased filter capacity (up to 256 filters)

Better feature learning and generalization

🧠 Improved CNN Architecture

Layer Flow:

Conv → BatchNorm → ReLU → MaxPool   (×4 blocks)
Dense → BatchNorm → ReLU → Dropout
Softmax Output

Why this design is better

Batch Normalization stabilizes gradients and speeds up training

Dropout reduces co-adaptation of neurons and prevents overfitting

Deeper architecture captures richer hierarchical audio features

Higher filter capacity learns complex timbral patterns of instruments

📂 Folder Structure
cnn_improvement/
│
├── model_definition.py     # Improved deep CNN architecture
├── train_evaluate.py       # Training, validation & evaluation
├── inference.py            # Model inference (prediction)
│
├── models/                 # Saved best model (ignored in git)
└── results/                # Metrics, plots, confusion matrix

📈 Performance Comparison
Model	Accuracy	F1 Score
Baseline CNN	~0.91	~0.92
Improved CNN	~0.93+	~0.96
✅ Which model is better?

The Improved CNN is clearly better.

🔍 Why the Improved CNN performs better:

Learns deeper and more discriminative audio features

Generalizes better on unseen data

Reduced validation loss fluctuation

Higher Precision, Recall, and F1 Score

Less overfitting due to BatchNorm + Dropout


Confusion Matrix

Saves plots and logs for analysis

🔍 Inference Module
inference.py
Why is inference.py important?

Training alone does not prove real-world usability.

inference.py:

Loads the trained improved CNN model

Accepts a Mel spectrogram input

Predicts the instrument class

Outputs prediction confidence

📌 This demonstrates that the model is:

Deployable

Ready for real-time or batch prediction

Not just an academic training experiment