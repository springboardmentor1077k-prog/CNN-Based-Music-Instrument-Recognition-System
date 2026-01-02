import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = "IRMAS-TrainingData"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# AUDIO PARAMETERS
# -------------------------------
SR = 22050
DURATION = 3
N_MELS = 128
MAX_LEN = 130
EPOCHS = 25
BATCH_SIZE = 32
LR = 0.0001

# -------------------------------
# LOAD DATASET
# -------------------------------
def load_dataset(path):
    X, y = [], []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".wav"):
                labels = file.replace(".wav", "").split("_")[2:]
                audio, _ = librosa.load(os.path.join(root, file), sr=SR, duration=DURATION)
                mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS)
                mel_db = librosa.power_to_db(mel, ref=np.max)

                if mel_db.shape[1] < MAX_LEN:
                    mel_db = np.pad(mel_db, ((0,0),(0,MAX_LEN-mel_db.shape[1])))
                else:
                    mel_db = mel_db[:, :MAX_LEN]

                X.append(mel_db)
                y.append(labels)

    return np.array(X), y

X, y = load_dataset(DATASET_PATH)
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)
X = X[..., np.newaxis]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# MODEL DEFINITION
# -------------------------------
def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=X_train.shape[1:]),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.1),

        Conv2D(64, (3,3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.15),

        Conv2D(128, (3,3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.2),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(y_train.shape[1], activation="sigmoid")
    ])
    return model

optimizers = {
    "SGD": SGD(learning_rate=LR),
    "Adam": Adam(learning_rate=LR),
    "RMSprop": RMSprop(learning_rate=LR)
}

histories = {}

# -------------------------------
# TRAIN WITH EACH OPTIMIZER
# -------------------------------
for name, opt in optimizers.items():
    print(f"\nTraining with {name}")
    model = build_model()
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    histories[name] = history.history
    np.save(os.path.join(OUTPUT_DIR, f"{name.lower()}_history.npy"), history.history)

# -------------------------------
# PLOT COMPARISON
# -------------------------------
plt.figure(figsize=(10,6))
for name in histories:
    plt.plot(histories[name]["val_loss"], label=f"{name} Val Loss")

plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.title("Optimizer Comparison – Validation Loss")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "optimizer_comparison.png"))
plt.close()
