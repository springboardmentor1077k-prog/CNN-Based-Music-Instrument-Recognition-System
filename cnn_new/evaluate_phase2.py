import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# =========================
# PATHS
# =========================
BASE_DIR = r"C:\Users\ADMIN\Downloads\check\cnn_new"

FEATURE_DIR = os.path.join(BASE_DIR, "features")
DATA_DIR = os.path.join(BASE_DIR, "data", "audio_csv")

MODEL_PATH = os.path.join(BASE_DIR, "model_instrunet_phase2.keras")

TEST_X_PATH = os.path.join(FEATURE_DIR, "X_test.npy")
TEST_Y_PATH = os.path.join(FEATURE_DIR, "y_test.npy")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")

# =========================
# LOAD CLASS NAMES (FROM CSV)
# =========================
print("📦 Loading class names from train.csv...")
train_df = pd.read_csv(TRAIN_CSV)
class_names = sorted(train_df["label"].unique())
num_classes = len(class_names)

print(f"Classes ({num_classes}):")
for i, name in enumerate(class_names):
    print(f"{i:02d} → {name}")

# =========================
# LOAD DATA
# =========================
print("\n📦 Loading test data...")
X_test = np.load(TEST_X_PATH)
y_test_raw = np.load(TEST_Y_PATH)

# Convert string labels → integer indices
label_to_index = {name: idx for idx, name in enumerate(class_names)}
y_test = np.array([label_to_index[lbl] for lbl in y_test_raw])

# =========================
# LOAD MODEL
# =========================
print("\n🧠 Loading Phase-2 model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded")

# =========================
# PREDICTION
# =========================
print("\n🔮 Running inference...")
y_probs = model.predict(X_test, batch_size=32)
y_pred = np.argmax(y_probs, axis=1)

# =========================
# METRICS
# =========================
acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")

print("\n📊 PHASE-2 TEST METRICS")
print(f"Accuracy  : {acc:.4f}")
print(f"Macro F1  : {f1_macro:.4f}")

print("\n📄 Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    digits=4
))

# =========================
# CONFUSION MATRIX (NO SEABORN)
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(14, 12))
plt.imshow(cm, cmap="Blues")
plt.colorbar()

plt.xticks(range(num_classes), class_names, rotation=90)
plt.yticks(range(num_classes), class_names)

plt.title("Phase-2 Confusion Matrix — InstruNetCNN")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
cm_path = os.path.join(BASE_DIR, "confusion_matrix_phase2.png")
plt.savefig(cm_path, dpi=300)
plt.show()

print(f"\n✅ Confusion matrix saved at:\n{cm_path}")
