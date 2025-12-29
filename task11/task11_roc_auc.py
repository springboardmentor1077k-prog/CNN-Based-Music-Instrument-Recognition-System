import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import load_model

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

MODEL_PATH = os.path.join(BASE_DIR, "task11_multilabel_model.h5")
Y_TRUE_PATH = os.path.join(OUTPUT_DIR, "y_true.npy")
Y_PROB_PATH = os.path.join(OUTPUT_DIR, "y_probabilities.npy")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# LOAD DATA
# -------------------------------
model = load_model(MODEL_PATH)
y_true = np.load(Y_TRUE_PATH)
y_prob = np.load(Y_PROB_PATH)

n_classes = y_true.shape[1]

# -------------------------------
# ROC & AUC
# -------------------------------
fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Micro-average ROC
fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_prob.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# -------------------------------
# PLOT ROC
# -------------------------------
plt.figure(figsize=(8, 6))
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label=f"Micro-average ROC (AUC = {roc_auc['micro']:.2f})",
    linewidth=2
)

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Label ROC Curve")
plt.legend(loc="lower right")

plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
plt.close()

# -------------------------------
# SAVE AUC SCORES
# -------------------------------
with open(os.path.join(OUTPUT_DIR, "auc_scores.txt"), "w") as f:
    f.write(f"Micro-average AUC: {roc_auc['micro']:.4f}\n\n")
    for i in range(n_classes):
        f.write(f"Class {i} AUC: {roc_auc[i]:.4f}\n")

print("ROC curve and AUC scores saved successfully.")
