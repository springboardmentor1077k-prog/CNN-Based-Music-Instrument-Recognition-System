# =====================================================
# Optimizer Experiments with Callbacks (Adam, SGD, RMS)
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
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.utils import to_categorical

# =====================================================
# LOAD DATA (FROM YOUR PIPELINE)
# =====================================================

X = np.load("cnn_features/X_mel.npy")      # (N, 128, 128, 1)
y = np.load("cnn_features/y_labels.npy")  # (N,)

num_classes = len(np.unique(y))

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", num_classes)

# Train / Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)

# =====================================================
# CNN MODEL DEFINITION (SAME FOR ALL OPTIMIZERS)
# =====================================================

def build_cnn_model(optimizer, name):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 1)),
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

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"\nModel compiled with {name}")
    return model

# =====================================================
# CALLBACKS (USED FOR ALL RUNS)
# =====================================================

def get_callbacks(name):
    return [
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
        ),
        ModelCheckpoint(
            filepath=f"{name}_best_model.h5",
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

# =====================================================
# OPTIMIZER EXPERIMENTS
# =====================================================

optimizers = {
    "Adam": Adam(learning_rate=0.0003),
    "SGD": SGD(learning_rate=0.01, momentum=0.9),
    "RMSprop": RMSprop(learning_rate=0.0003)
}

histories = {}

for name, optimizer in optimizers.items():
    print(f"\n===== Training with {name} optimizer =====")

    model = build_cnn_model(optimizer, name)
    callbacks = get_callbacks(name)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    histories[name] = history

# =====================================================
# PLOT COMPARISON RESULTS
# =====================================================

plt.figure(figsize=(14, 6))

# Accuracy
plt.subplot(1, 2, 1)
for name, history in histories.items():
    plt.plot(history.history["val_accuracy"], label=name)
plt.title("Validation Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
for name, history in histories.items():
    plt.plot(history.history["val_loss"], label=name)
plt.title("Validation Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

print("\nOptimizer experiments completed.")
print("Best models saved as:")
print("- Adam_best_model.h5")
print("- SGD_best_model.h5")
print("- RMSprop_best_model.h5")
