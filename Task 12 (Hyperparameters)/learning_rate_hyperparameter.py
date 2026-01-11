# =====================================================
# Single-Label vs Multi-Label CNN Training (Audio)
# =====================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    BatchNormalization,
    Dense, Dropout, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# =====================================================
# LOAD PREPROCESSED FEATURES (FROM TASK 8)
# =====================================================

X = np.load("cnn_features/X_mel.npy")      # (N, 128, 128, 1)
y = np.load("cnn_features/y_labels.npy")  # (N,)

print("Loaded data:")
print("X shape:", X.shape)
print("y shape:", y.shape)

num_classes = len(np.unique(y))
input_shape = X.shape[1:]

# =====================================================
# TRAIN / VALIDATION SPLIT
# =====================================================

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# ================== SINGLE-LABEL TRAINING =============
# =====================================================

print("\nStarting SINGLE-LABEL training...")

# One-hot encoding
y_train_sl = to_categorical(y_train, num_classes)
y_val_sl = to_categorical(y_val, num_classes)

# -------- CNN Model (Single Label) --------
single_label_model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),

    Dense(num_classes, activation="softmax")
])

single_label_model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

single_label_model.summary()

history_sl = single_label_model.fit(
    X_train, y_train_sl,
    validation_data=(X_val, y_val_sl),
    epochs=20,
    batch_size=32
)

# Save model
single_label_model.save("single_label_cnn.h5")

# =====================================================
# ================= MULTI-LABEL TRAINING ==============
# =====================================================

print("\nStarting MULTI-LABEL training (simulated)...")

# -------- Create pseudo multi-label targets --------
# NOTE: IRMAS is single-label, so we simulate multi-label data

np.random.seed(42)
y_multilabel = np.zeros((len(y), num_classes))

for i, label in enumerate(y):
    y_multilabel[i, label] = 1
    # Randomly add another instrument (simulation)
    if np.random.rand() > 0.7:
        random_label = np.random.randint(0, num_classes)
        y_multilabel[i, random_label] = 1

X_train_ml, X_val_ml, y_train_ml, y_val_ml = train_test_split(
    X, y_multilabel,
    test_size=0.2,
    random_state=42
)

# -------- CNN Model (Multi Label) --------
multi_label_model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),

    Dense(num_classes, activation="sigmoid")
])

multi_label_model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

multi_label_model.summary()

history_ml = multi_label_model.fit(
    X_train_ml, y_train_ml,
    validation_data=(X_val_ml, y_val_ml),
    epochs=20,
    batch_size=32
)

# Save model
multi_label_model.save("multi_label_cnn.h5")

# =====================================================
# ================= TRAINING CURVES ===================
# =====================================================

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history_sl.history["accuracy"], label="Single-label Train")
plt.plot(history_sl.history["val_accuracy"], label="Single-label Val")
plt.plot(history_ml.history["accuracy"], label="Multi-label Train")
plt.plot(history_ml.history["val_accuracy"], label="Multi-label Val")
plt.title("Accuracy Comparison")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history_sl.history["loss"], label="Single-label Train")
plt.plot(history_sl.history["val_loss"], label="Single-label Val")
plt.plot(history_ml.history["loss"], label="Multi-label Train")
plt.plot(history_ml.history["val_loss"], label="Multi-label Val")
plt.title("Loss Comparison")
plt.legend()

plt.tight_layout()
plt.show()

print("\nTraining completed.")
print("Models saved:")
print("- single_label_cnn.h5")
print("- multi_label_cnn.h5")
