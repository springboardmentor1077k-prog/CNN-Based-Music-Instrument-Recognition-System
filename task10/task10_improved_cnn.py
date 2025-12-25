"""
Task 10 – Improved CNN Model for Music Instrument Recognition
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten,
    Dropout, BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =========================
# DATASET PATH (FIXED)
# =========================
DATASET_PATH = "IRMAS-TrainingData\IRMAS-TrainingData"

# =========================
# OUTPUT DIRECTORY
# =========================
OUTPUT_DIR = "task10"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# AUDIO PARAMETERS
# =========================
SAMPLE_RATE = 22050
DURATION = 3
N_MELS = 128
MAX_LEN = 130

# =========================
# LOAD DATASET (IRMAS SAFE)
# =========================
def load_dataset(dataset_path):
    X, y = [], []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if not file.endswith(".wav"):
                continue

            file_path = os.path.join(root, file)

            # Extract instrument label from filename
            # Example: Train_001_Clarinet.wav → Clarinet
            instrument = file.split("_")[-1].replace(".wav", "")

            try:
                signal, sr = librosa.load(
                    file_path,
                    sr=SAMPLE_RATE,
                    duration=DURATION
                )

                mel_spec = librosa.feature.melspectrogram(
                    y=signal,
                    sr=sr,
                    n_mels=N_MELS
                )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

                if mel_spec_db.shape[1] < MAX_LEN:
                    pad_width = MAX_LEN - mel_spec_db.shape[1]
                    mel_spec_db = np.pad(
                        mel_spec_db,
                        ((0, 0), (0, pad_width))
                    )
                else:
                    mel_spec_db = mel_spec_db[:, :MAX_LEN]

                X.append(mel_spec_db)
                y.append(instrument)

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    return np.array(X), np.array(y)

# =========================
# LOAD DATA
# =========================
print("Loading dataset...")
X, y = load_dataset(DATASET_PATH)

print("Total samples loaded:", len(X))
print("Unique instruments:", np.unique(y))

# =========================
# ENCODE LABELS
# =========================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_encoded = to_categorical(y_encoded)

# =========================
# RESHAPE FOR CNN
# =========================
X = X[..., np.newaxis]

# =========================
# TRAIN / VALIDATION SPLIT
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape)
print("Validation samples:", X_val.shape)

# =========================
# IMPROVED CNN MODEL (TASK 10)
# =========================
model = Sequential()

# Block 1
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.1))

# Block 2
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.15))

# Block 3
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

# Dense layers
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(y_train.shape[1], activation="softmax"))

model.summary()

# =========================
# COMPILE
# =========================
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# TRAIN
# =========================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=32
)

# =========================
# SAVE MODEL
# =========================
model.save(os.path.join(OUTPUT_DIR, "model_task10.h5"))
print("Model saved as model_task10.h5")

# =========================
# PLOTS
# =========================
plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.title("Accuracy Curve - Task 10")
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"))
plt.close()

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss Curve - Task 10")
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
plt.close()

print("Accuracy and loss curves saved.")
