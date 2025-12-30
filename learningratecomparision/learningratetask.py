import numpy as np
import tensorflow as tf
from PIL import Image

# ---------------- CONFIG ----------------
IMG_SIZE = (128, 128)
EPOCHS = 5
BATCH_SIZE = 2

SAMPLES = [
    ("pipeline/mel/test/87.png", 0),        # clean
    ("pipeline/mel/train/19_aug1.png", 1)   # augmented
]


def load_data():
    X, y = [], []
    for path, label in SAMPLES:
        img = Image.open(path).convert("L").resize(IMG_SIZE)
        x = np.array(img) / 255.0
        X.append(x)
        y.append(label)
    X = np.array(X).reshape(-1, 128, 128, 1)
    y = tf.keras.utils.to_categorical(y, 2)
    return X, y

def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, 3, activation="relu", input_shape=(128,128,1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax")
    ])
    return model

def run_experiment(learning_rate):
    print(f"\n🚀 Training with learning rate = {learning_rate}")
    
    X, y = load_data()
    model = build_model()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    preds = model.predict(X)
    
    print("Predictions:")
    for i, (path, _) in enumerate(SAMPLES):
        print(f"{path} → probs = {preds[i]}")

    return preds

# -------- RUN BOTH EXPERIMENTS --------
preds_lr_high = run_experiment(learning_rate=1e-3)
preds_lr_low  = run_experiment(learning_rate=1e-4)
