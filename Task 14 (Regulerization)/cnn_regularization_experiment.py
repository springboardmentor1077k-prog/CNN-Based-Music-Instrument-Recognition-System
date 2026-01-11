# =====================================================
# CNN Regularization Implementation (Audio Classification)
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
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# =====================================================
# LOAD DATA (FROM PREVIOUS PIPELINE)
# =====================================================

X = np.load("cnn_features/X_mel.npy")      # (N, 128, 128, 1)
y = np.load("cnn_features/y_labels.npy")  # (N,)

num_classes = len(np.unique(y))

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", num_classes)

# =====================================================
# TRAIN / VALIDATION SPLIT
# =====================================================

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# One-hot encoding
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)

# =====================================================
# REGULARIZED CNN MODEL
# =====================================================

model = Sequential([

    # ----- Convolution Block 1 -----
    Conv2D(
        32, (3, 3),
        activation="relu",
        input_shape=(128, 128, 1),
        kernel_regularizer=l2(0.001)
    ),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # ----- Convolution Block 2 -----
    Conv2D(
        64, (3, 3),
        activation="relu",
        kernel_regularizer=l2(0.001)
    ),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # ----- Convolution Block 3 -----
    Conv2D(
        128, (3, 3),
        activation="relu",
        kernel_regularizer=l2(0.001)
    ),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),

    # ----- Fully Connected Layer -----
    Dense(
        256,
        activation="relu",
        kernel_regularizer=l2(0.001)
    ),
    Dropout(0.5),

    # ----- Output Layer -----
    Dense(num_classes, activation="softmax")
])

# =====================================================
# COMPILE MODEL
# =====================================================

model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================================================
# CALLBACKS
# =====================================================

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1
    )
]

# =====================================================
# TRAIN MODEL
# =====================================================

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# =====================================================
# PLOT TRAINING CURVES
# =====================================================

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy with Regularization")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss with Regularization")
plt.legend()

plt.tight_layout()
plt.show()

# =====================================================
# SAVE MODEL
# =====================================================

model.save("cnn_regularized_model.h5")
print("Regularized CNN model saved successfully")
