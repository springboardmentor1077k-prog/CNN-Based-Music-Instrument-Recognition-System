import os
import numpy as np
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = "IRMAS-TrainingData"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "task11_multilabel_model.h5")

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

            file_path = os.path.join(root, file)

            # Example: Train_001_Guitar_Piano.wav
            labels = file.replace(".wav", "").split("_")[2:]

            try:
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
                mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=N_MELS)
                mel_db = librosa.power_to_db(mel, ref=np.max)

                if mel_db.shape[1] < MAX_LEN:
                    mel_db = np.pad(mel_db, ((0, 0), (0, MAX_LEN - mel_db.shape[1])))
                else:
                    mel_db = mel_db[:, :MAX_LEN]

                X.append(mel_db)
                y.append(labels)

            except Exception as e:
                print("Error:", e)

    return np.array(X), y

# -------------------------------
# LOAD DATA
# -------------------------------
X, y = load_dataset(DATASET_PATH)

print("Checking first 20 samples:")
for labels in y[:20]:
    print(labels, " -> count:", len(labels))

# Multi-label encoding
mlb = MultiLabelBinarizer()
y_encoded = mlb.fit_transform(y)

# Expand dims for CNN
X = X[..., np.newaxis]

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# -------------------------------
# MODEL
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
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(y_train.shape[1], activation='sigmoid')
])

model.compile(
    optimizer=Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------
# TRAIN
# -------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=32
)

# -------------------------------
# EVALUATION
# -------------------------------
y_pred = model.predict(X_val)
y_pred_bin = (y_pred > 0.5).astype(int)

# Save predictions
np.save(os.path.join(OUTPUT_DIR, "y_probabilities.npy"), y_pred)
np.save(os.path.join(OUTPUT_DIR, "y_pred_binary.npy"), y_pred_bin)
np.save(os.path.join(OUTPUT_DIR, "y_true.npy"), y_val)

# Classification report
report = classification_report(y_val, y_pred_bin, target_names=mlb.classes_)
print("\nClassification Report:\n", report)

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# Confusion matrix (class-wise dominant label)
cm = confusion_matrix(y_val.argmax(axis=1), y_pred_bin.argmax(axis=1))
np.save(os.path.join(OUTPUT_DIR, "confusion_matrix.npy"), cm)

# Metrics
precision = precision_score(y_val, y_pred_bin, average="micro")
recall = recall_score(y_val, y_pred_bin, average="micro")
f1 = f1_score(y_val, y_pred_bin, average="micro")

with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")

# -------------------------------
# SAVE MODEL
# -------------------------------
model.save(MODEL_PATH)
print(f"\nModel saved at: {MODEL_PATH}")
