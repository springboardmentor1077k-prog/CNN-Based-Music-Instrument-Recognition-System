import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# ================= CONFIG =================
PIPELINE_DIR = "../pipeline"
MEL_DIR = os.path.join(PIPELINE_DIR, "mel")
META_DIR = os.path.join(PIPELINE_DIR, "metadata")

IMG_H = 128
IMG_W = 128
CHANNELS = 1
RANDOM_SEED = 42
# =========================================


print("Loading metadata...")

train_df = pd.read_csv(os.path.join(META_DIR, "train.csv"))
val_df   = pd.read_csv(os.path.join(META_DIR, "valid.csv"))
test_df  = pd.read_csv(os.path.join(META_DIR, "test.csv"))

print("Train CSV columns:", train_df.columns.tolist())

# ---------- LABEL ENCODING ----------
label_encoder = LabelEncoder()
train_df["label_id"] = label_encoder.fit_transform(train_df["type"])
val_df["label_id"]   = label_encoder.transform(val_df["type"])
test_df["label_id"]  = label_encoder.transform(test_df["type"])

NUM_CLASSES = len(label_encoder.classes_)
print("Number of classes:", NUM_CLASSES)


# ---------- LOAD PNG MEL ----------
def load_png_mel(path):
    img = Image.open(path).convert("L")  # grayscale
    img = img.resize((IMG_W, IMG_H))
    img = np.array(img, dtype=np.float32) / 255.0
    img = img[..., np.newaxis]           # (H, W, 1)
    return img


def load_split(df, split_name):
    X, y = [], []
    split_dir = os.path.join(MEL_DIR, split_name)

    print(f"\nLoading {split_name} mel spectrograms from:", split_dir)

    for _, row in df.iterrows():
        fname = os.path.splitext(row["filename"])[0] + ".png"
        mel_path = os.path.join(split_dir, fname)

        if not os.path.exists(mel_path):
            continue

        X.append(load_png_mel(mel_path))
        y.append(row["label_id"])

    return np.array(X), tf.keras.utils.to_categorical(y, NUM_CLASSES)


X_train, y_train = load_split(train_df, "train")
X_val,   y_val   = load_split(val_df, "valid")
X_test,  y_test  = load_split(test_df, "test")


# ---------- CNN ARCHITECTURE ----------
def build_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, 3, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),

        layers.Dense(num_classes, activation="softmax")
    ])
    return model


model = build_cnn(
    input_shape=(IMG_H, IMG_W, CHANNELS),
    num_classes=NUM_CLASSES
)

model.summary()

print("\nSanity Check:")
print("X_train shape:", X_train.shape)
print("X_val shape  :", X_val.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_val shape  :", y_val.shape)
print("y_test shape :", y_test.shape)
