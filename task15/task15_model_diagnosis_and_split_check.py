import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# -------------------------------------------------
# STEP 1: Freeze Validation Set (from Task 9)
# -------------------------------------------------

OUTPUT_DIR = "task15/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

X_val = np.load("task9/outputs_task9/X_val.npy")
y_val = np.load("task9/outputs_task9/y_val.npy")

np.save(os.path.join(OUTPUT_DIR, "X_val_frozen.npy"), X_val)
np.save(os.path.join(OUTPUT_DIR, "y_val_frozen.npy"), y_val)

print("✅ Validation set frozen and saved")

# -------------------------------------------------
# STEP 2: Load Trained Model (Task 9 / 14)
# -------------------------------------------------

model = load_model("task9/outputs_task9/cnn_model.h5")
print("✅ Model loaded for evaluation")

# -------------------------------------------------
# STEP 3: Evaluate on Frozen Validation Set
# -------------------------------------------------

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

# -------------------------------------------------
# STEP 4: Manual Diagnosis (Based on Previous Tasks)
# -------------------------------------------------

print("\nModel Diagnosis:")
print("- Training accuracy was higher than validation accuracy")
print("- Gap reduced after Dropout and L2 regularization")
print("- Validation accuracy is stable")
print("Diagnosis: Healthy / Controlled Overfitting")
