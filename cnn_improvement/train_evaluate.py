import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import optimizers, callbacks
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

from model_definition import build_deep_cnn


# =========================
# PATH CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PIPELINE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "pipeline"))
META_DIR = os.path.join(PIPELINE_DIR, "metadata")

TRAIN_CSV = os.path.join(META_DIR, "train.csv")
VAL_CSV   = os.path.join(META_DIR, "valid.csv")
TEST_CSV  = os.path.join(META_DIR, "test.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# =========================
# PARAMETERS
# =========================
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50


# =========================
# IMAGE LOADER
# =========================
def load_mel_image(img_path):
    if not os.path.exists(img_path):
        return None

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    return img


# =========================
# LOAD SPLIT USING mel_path
# =========================
def load_split(df):
    X, y = [], []

    for _, row in df.iterrows():
        img_path = row["mel_path"]  # ✅ FIXED
        img = load_mel_image(img_path)

        if img is None:
            continue

        X.append(img)
        y.append(row["label_id"])

    if len(X) == 0:
        return np.array([]), np.array([])

    X = np.array(X, dtype=np.float32)
    y = tf.keras.utils.to_categorical(y)

    return X, y


# =========================
# LOAD METADATA
# =========================
print("Loading metadata...")

train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)
test_df  = pd.read_csv(TEST_CSV)

print("Train CSV columns:", train_df.columns.tolist())

label_encoder = LabelEncoder()
train_df["label_id"] = label_encoder.fit_transform(train_df["type"])
val_df["label_id"]   = label_encoder.transform(val_df["type"])
test_df["label_id"]  = label_encoder.transform(test_df["type"])

NUM_CLASSES = len(label_encoder.classes_)
print("Number of classes:", NUM_CLASSES)


# =========================
# LOAD DATA
# =========================
print("\nLoading train data...")
X_train, y_train = load_split(train_df)

print("Loading validation data...")
X_val, y_val = load_split(val_df)

print("Loading test data...")
X_test, y_test = load_split(test_df)


# =========================
# SANITY CHECK
# =========================
print("\nSanity Check:")
print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)

if X_train.size == 0:
    raise RuntimeError("❌ No training data loaded. Check mel_path values in CSV!")


# =========================
# MODEL
# =========================
model = build_deep_cnn(
    input_shape=(IMG_SIZE, IMG_SIZE, 1),
    num_classes=NUM_CLASSES
)

model.compile(
    optimizer=optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =========================
# CALLBACKS
# =========================
checkpoint = callbacks.ModelCheckpoint(
    os.path.join(MODEL_DIR, "deep_cnn_best.h5"),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)


# =========================
# TRAIN
# =========================
print("\n🚀 Training deep CNN...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, early_stop, reduce_lr]
)


# =========================
# EVALUATE
# =========================
print("\nEvaluating on test set...")
y_pred = model.predict(X_test)
y_pred_cls = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

acc = accuracy_score(y_true, y_pred_cls)
report = classification_report(y_true, y_pred_cls, target_names=label_encoder.classes_)

print("\nTest Accuracy:", acc)
print("\nClassification Report:\n", report)

with open(os.path.join(RESULT_DIR, "test_report.txt"), "w") as f:
    f.write(f"Accuracy: {acc}\n\n")
    f.write(report)

print("\n✅ Training complete.")
print("✅ Model saved at:", os.path.join(MODEL_DIR, "deep_cnn_best.h5"))
