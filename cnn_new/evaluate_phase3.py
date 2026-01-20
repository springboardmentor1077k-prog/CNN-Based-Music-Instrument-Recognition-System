import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE_DIR = r"C:\Users\ADMIN\Downloads\check\cnn_new"

RESULTS_PATH = f"{BASE_DIR}/phase3_results.npy"
CLASSES_PATH = f"{BASE_DIR}/phase3_classes.npy"

print("📦 Loading Phase-3 aggregation results...")
results = np.load(RESULTS_PATH, allow_pickle=True).item()

print("📦 Loading class names...")
class_names = np.load(CLASSES_PATH, allow_pickle=True)

# Containers
y_true = []
y_mean = []
y_max = []
y_vote = []

for audio_path, preds in results.items():
    y_true.append(preds["true"])
    y_mean.append(preds["mean"])
    y_max.append(preds["max"])
    y_vote.append(preds["vote"])

y_true = np.array(y_true)
y_mean = np.array(y_mean)
y_max = np.array(y_max)
y_vote = np.array(y_vote)

# =========================
# METRICS
# =========================
def evaluate(name, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n📊 {name.upper()} AGGREGATION")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {f1:.4f}")
    return acc, f1

print("\n==============================")
print("🎧 PHASE-3 AUDIO-LEVEL RESULTS")
print("==============================")

mean_acc, mean_f1 = evaluate("Mean", y_mean)
max_acc, max_f1 = evaluate("Max", y_max)
vote_acc, vote_f1 = evaluate("Voting", y_vote)

# =========================
# BEST METHOD
# =========================
scores = {
    "Mean": mean_f1,
    "Max": max_f1,
    "Voting": vote_f1
}

best_method = max(scores, key=scores.get)

print("\n🏆 BEST AGGREGATION METHOD")
print(f"Method  : {best_method}")
print(f"Macro F1: {scores[best_method]:.4f}")

# =========================
# OPTIONAL: Detailed report
# =========================
print("\n📄 Classification Report (BEST METHOD)")
best_preds = {
    "Mean": y_mean,
    "Max": y_max,
    "Voting": y_vote
}[best_method]

print(classification_report(
    y_true,
    best_preds,
    target_names=class_names
))
