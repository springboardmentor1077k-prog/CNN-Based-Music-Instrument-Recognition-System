import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from instrunet_cnn import InstruNetCNN

# =========================
# PATHS
# =========================
BASE_DIR = r"C:\Users\ADMIN\Downloads\check\cnn_new"
FEATURE_DIR = os.path.join(BASE_DIR, "features")

MODEL_SAVE_PATH = os.path.join(BASE_DIR, "model_instrunet_phase2.keras")

# =========================
# LOAD FEATURES
# =========================
X_train = np.load(os.path.join(FEATURE_DIR, "X_train.npy"))
y_train_raw = np.load(os.path.join(FEATURE_DIR, "y_train.npy"), allow_pickle=True)

X_val = np.load(os.path.join(FEATURE_DIR, "X_val.npy"))
y_val_raw = np.load(os.path.join(FEATURE_DIR, "y_val.npy"), allow_pickle=True)

print("\n📊 DATA SUMMARY (RAW)")
print("Train:", X_train.shape, " Val:", X_val.shape)
print("Sample labels:", np.unique(y_train_raw)[:5])

# =========================
# LABEL ENCODING
# =========================
label_encoder = LabelEncoder()
y_train_int = label_encoder.fit_transform(y_train_raw)
y_val_int = label_encoder.transform(y_val_raw)

num_classes = len(label_encoder.classes_)

print("\n🔢 Classes:", num_classes)
print("Class mapping:")
for i, cls in enumerate(label_encoder.classes_):
    print(f"{i:02d} → {cls}")

# =========================
# ONE-HOT ENCODING (FIX)
# =========================
y_train = tf.keras.utils.to_categorical(y_train_int, num_classes)
y_val = tf.keras.utils.to_categorical(y_val_int, num_classes)

# =========================
# CLASS WEIGHTS
# =========================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=y_train_int
)

class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}

print("\n⚖️ Class weights:", class_weight_dict)

# =========================
# BUILD MODEL
# =========================
instrunet = InstruNetCNN(
    input_shape=(128, 128, 1),
    num_classes=num_classes
)

instrunet.build_model()

instrunet.model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n🧠 MODEL SUMMARY")
instrunet.model.summary()

# =========================
# CALLBACKS (IMPORTANT)
# =========================
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# =========================
# TRAINING
# =========================
EPOCHS = 50
BATCH_SIZE = 32

print("\n🚀 STARTING PHASE-2 TRAINING (InstruNetCNN)")

history = instrunet.model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# =========================
# SAVE MODEL
# =========================
instrunet.model.save(MODEL_SAVE_PATH)
print(f"\n✅ Phase-2 model saved at: {MODEL_SAVE_PATH}")
