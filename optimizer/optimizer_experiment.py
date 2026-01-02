import os
import sys
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras import optimizers, callbacks

# -------------------------------------------------
# FIX PATH so cnn_improvement is importable
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from cnn_improvement.model_definition import build_deep_cnn

# -------------------------------------------------
# PATHS
# -------------------------------------------------
META_DIR = os.path.join(BASE_DIR, "pipeline", "metadata")
MEL_DIR  = os.path.join(BASE_DIR, "pipeline", "mel")

TRAIN_CSV = os.path.join(META_DIR, "train.csv")
VAL_CSV   = os.path.join(META_DIR, "valid.csv")

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10   # small epochs for comparison (fair + fast)

# -------------------------------------------------
# LOAD MEL IMAGE
# -------------------------------------------------
def load_mel_image(path):
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return img[..., np.newaxis]

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
def load_split(csv_path, split):
    df = pd.read_csv(csv_path)

    X, y = [], []
    for _, row in df.iterrows():
        fname = os.path.splitext(os.path.basename(row["filepath"]))[0]
        mel_path = os.path.join(MEL_DIR, split, f"{fname}.png")

        img = load_mel_image(mel_path)
        if img is None:
            continue

        X.append(img)
        y.append(row["label"])

    X = np.array(X, dtype=np.float32)

    le = LabelEncoder()
    y = le.fit_transform(y)
    y = tf.keras.utils.to_categorical(y)

    return X, y

# -------------------------------------------------
# TRAIN + EVALUATE
# -------------------------------------------------
def run_optimizer_experiment(optimizer, name):
    print(f"\n🚀 Optimizer: {name}")

    X_train, y_train = load_split(TRAIN_CSV, "train")
    X_val, y_val     = load_split(VAL_CSV, "valid")

    model = build_deep_cnn(
        input_shape=(IMG_SIZE, IMG_SIZE, 1),
        num_classes=2
    )

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es],
        verbose=0
    )

    y_prob = model.predict(X_val)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_val, axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_macro = f1_score(y_true, y_pred, average="macro")

    print(f"Accuracy  : {acc:.4f}")
    print(f"F1 Micro  : {f1_micro:.4f}")
    print(f"F1 Macro  : {f1_macro:.4f}")

    return acc, f1_micro, f1_macro

# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":

    results = {}

    results["SGD"] = run_optimizer_experiment(
        optimizers.SGD(learning_rate=0.01, momentum=0.9),
        "SGD"
    )

    results["RMSprop"] = run_optimizer_experiment(
        optimizers.RMSprop(learning_rate=0.001),
        "RMSprop"
    )

    results["Adam"] = run_optimizer_experiment(
        optimizers.Adam(learning_rate=0.001),
        "Adam"
    )

    print("\n📊 FINAL COMPARISON")
    print("=" * 40)
    for k, v in results.items():
        print(f"{k:8s} → Acc={v[0]:.4f}, F1-micro={v[1]:.4f}, F1-macro={v[2]:.4f}")
