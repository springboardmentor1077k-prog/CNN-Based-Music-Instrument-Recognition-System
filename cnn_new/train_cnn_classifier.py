import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

DATA_DIR = "cnn_new/data"
MEL_DIR = f"{DATA_DIR}/mel"

train_df = pd.read_csv(f"{DATA_DIR}/audio_csv/train.csv")
val_df = pd.read_csv(f"{DATA_DIR}/audio_csv/val.csv")
test_df = pd.read_csv(f"{DATA_DIR}/audio_csv/test.csv")

le = LabelEncoder()
y_train = le.fit_transform(train_df.label)
y_val = le.transform(val_df.label)
y_test = le.transform(test_df.label)

def load_mels(df):
    X = []
    for p in df.path:
        name = os.path.basename(p).replace(".wav", ".npy")
        mel = np.load(f"{MEL_DIR}/{name}")
        mel = np.expand_dims(mel, -1)
        X.append(mel)
    return np.array(X)

X_train = load_mels(train_df)
X_val = load_mels(val_df)
X_test = load_mels(test_df)

num_classes = len(le.classes_)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, X_train.shape[2], 1)),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True
        )
    ]
)

model.save("cnn_new/model_instrunet.keras")

# Evaluation
y_pred = np.argmax(model.predict(X_test), axis=1)

print("\n📊 FINAL TEST METRICS")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred, average="macro"))
print("Precision:", precision_score(y_test, y_pred, average="macro"))
print("Recall   :", recall_score(y_test, y_pred, average="macro"))
