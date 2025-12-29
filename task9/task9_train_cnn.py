# task9/task9_train_cnn.py
# CNN training on FULL IRMAS dataset (all instruments)

import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Flatten, Dense, Dropout, Input
)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# -------------------------
# Paths
# -------------------------
DATASET_PATH = "IRMAS-TrainingData/IRMAS-TrainingData"
OUTPUT_DIR = "task9/outputs_task9"
MODEL_PATH = os.path.join(OUTPUT_DIR, "cnn_model.h5")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Parameters
# -------------------------
SR = 22050
N_MELS = 128
MAX_DURATION = 3.0  # seconds
MAX_LEN = int(SR * MAX_DURATION)

# -------------------------
# Load dataset
# -------------------------
X = []
y = []

classes = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])

print("Detected classes:", classes)
print("Number of classes:", len(classes))

for label, inst in enumerate(classes):
    inst_path = os.path.join(DATASET_PATH, inst)

    for file in os.listdir(inst_path):
        if file.endswith(".wav"):
            file_path = os.path.join(inst_path, file)

            try:
                audio, sr = librosa.load(file_path, sr=SR, mono=True)

                # Fix duration
                if len(audio) > MAX_LEN:
                    audio = audio[:MAX_LEN]
                else:
                    audio = np.pad(audio, (0, MAX_LEN - len(audio)))

                # Mel spectrogram
                mel = librosa.feature.melspectrogram(
                    y=audio, sr=sr, n_mels=N_MELS
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)

                X.append(mel_db)
                y.append(label)

            except Exception as e:
                print(f"Skipping {file_path}: {e}")

X = np.array(X)
y = np.array(y)

# Add channel dimension
X = X[..., np.newaxis]

# One-hot labels
y = to_categorical(y, num_classes=len(classes))

print("Data shape:", X.shape)
print("Labels shape:", y.shape)

# -------------------------
# Train / validation split
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Save validation data (for Task 11)
# -------------------------
np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)

# -------------------------
# CNN model (OWN DESIGN)
# -------------------------
model = Sequential([
    Input(shape=X_train.shape[1:]),

    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),

    Dense(len(classes), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------
# Summary (console)
# -------------------------
model.summary()

# -------------------------
# Train
# -------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=16
)

# -------------------------
# Save model
# -------------------------
model.save(MODEL_PATH)
print(f"\nModel saved at: {MODEL_PATH}")
