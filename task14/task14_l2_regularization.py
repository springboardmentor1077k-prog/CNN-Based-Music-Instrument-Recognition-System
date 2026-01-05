import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# -------------------------------
# PATHS
# -------------------------------
DATASET_PATH = "IRMAS-TrainingData"
OUTPUT_DIR = "task14/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# AUDIO PARAMETERS
# -------------------------------
SAMPLE_RATE = 22050
DURATION = 3
N_MELS = 128
MAX_LEN = 130

# -------------------------------
# LOAD DATASET (MULTI-LABEL)
# -------------------------------
def load_dataset(path):
    X, y = [], []

    for root, _, files in os.walk(path):
        for file in files:
            if not file.endswith(".wav"):
                continue

            labels = file.replace(".wav", "").split("_")[2:]
            file_path = os.path.join(root, file)

            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
            mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=N_MELS)
            mel_db = librosa.power_to_db(mel, ref=np.max)

            if mel_db.shape[1] < MAX_LEN:
                mel_db = np.pad(mel_db, ((0,0),(0, MAX_LEN - mel_db.shape[1])))
            else:
                mel_db = mel_db[:, :MAX_LEN]

            X.append(mel_db)
            y.append(labels)

    return np.array(X), y

# -------------------------------
# LOAD DATA
# -------------------------------
X, y = load_dataset(DATASET_PATH)

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)

X = X[..., np.newaxis]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# MODEL WITH L2 REGULARIZATION
# -------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=X_train.shape[1:]),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.1),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.15),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.2),

    Flatten(),

    # 🔑 L2 REGULARIZED DENSE LAYER
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),

    Dense(y_train.shape[1], activation='sigmoid')
])

model.compile(
    optimizer=Adam(),   # ❌ learning rate unchanged
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------
# TRAIN (SHORT RUN)
# -------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,   # short run
    batch_size=32
)

# -------------------------------
# SAVE MODEL
# -------------------------------
model.save("task14/outputs/task14_l2_model.h5")

# -------------------------------
# PLOT ACCURACY CURVES
# -------------------------------
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Task 14: Training vs Validation Accuracy (L2)")
plt.savefig("task14/outputs/task14_accuracy_curve.png")
plt.show()
