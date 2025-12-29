import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import os

# =========================
# PATHS
# =========================
MODEL_PATH = "task9/outputs_task9/cnn_model.h5"   # change if model path differs
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
X_val = np.load("task9/outputs_task9/X_val.npy")
y_val = np.load("task9/outputs_task9/y_val.npy")

model = tf.keras.models.load_model("task9/outputs_task9/cnn_model.h5")

# If labels are one-hot encoded, convert to class indices
if len(y_val.shape) > 1:
    y_true = np.argmax(y_val, axis=1)
else:
    y_true = y_val

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# PREDICTIONS
# =========================
y_prob = model.predict(X_val)

# Store raw probabilities
np.save(os.path.join(OUTPUT_DIR, "y_probabilities.npy"), y_prob)

# =========================
# CONVERT TO PREDICTIONS
# =========================
# Single-label IRMAS → argmax
y_pred = np.argmax(y_prob, axis=1)

# Convert to binary (one-hot)
num_classes = y_prob.shape[1]
y_pred_binary = np.zeros((len(y_pred), num_classes))
y_pred_binary[np.arange(len(y_pred)), y_pred] = 1

np.save(os.path.join(OUTPUT_DIR, "y_pred_binary.npy"), y_pred_binary)

# =========================
# METRICS
# =========================
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")
f1 = f1_score(y_true, y_pred, average="macro")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# Save metrics
with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)
np.save(os.path.join(OUTPUT_DIR, "confusion_matrix.npy"), cm)

print("\nConfusion Matrix:")
print(cm)

# =========================
# CLASSIFICATION REPORT
# =========================
report = classification_report(y_true, y_pred)
print("\nClassification Report:")
print(report)

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)
