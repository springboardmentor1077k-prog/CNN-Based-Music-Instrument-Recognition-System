import os
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(__file__)
FEATURE_DIR = os.path.join(BASE_DIR, "features")
DATA_DIR = os.path.join(BASE_DIR, "data", "audio_csv")

MODEL_PATH = os.path.join(BASE_DIR, "model_instrunet_phase2.keras")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

# Load features
X_test = np.load(os.path.join(FEATURE_DIR, "X_test.npy"))
y_test = np.load(os.path.join(FEATURE_DIR, "y_test.npy"))

df_test = pd.read_csv(TEST_CSV)

print("📄 CSV columns:", df_test.columns.tolist())

# 🔎 Auto-detect audio path column
PATH_COL = None
for col in ["filepath", "file_path", "path", "audio_path", "filename"]:
    if col in df_test.columns:
        PATH_COL = col
        break

if PATH_COL is None:
    raise ValueError("❌ Could not find audio path column in test.csv")

print(f"✅ Using audio path column: {PATH_COL}")

# Encode labels
le = LabelEncoder()
df_test["label_encoded"] = le.fit_transform(df_test["label"])

# Load model
print("\n🧠 Loading Phase-2 model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("🔮 Running segment-level predictions...")
segment_probs = model.predict(X_test, batch_size=32, verbose=1)
segment_preds = np.argmax(segment_probs, axis=1)

df_test["pred"] = segment_preds

# Aggregate per audio
audio_results = {}

for audio_path, group in df_test.groupby(PATH_COL):
    true_label = group["label_encoded"].iloc[0]

    probs = segment_probs[group.index]
    preds = group["pred"].values

    audio_results[audio_path] = {
        "true": true_label,
        "mean": np.argmax(probs.mean(axis=0)),
        "max": np.argmax(probs.max(axis=0)),
        "vote": Counter(preds).most_common(1)[0][0]
    }

# Save results
np.save(os.path.join(BASE_DIR, "phase3_results.npy"), audio_results)
np.save(os.path.join(BASE_DIR, "phase3_classes.npy"), le.classes_)

print("\n✅ PHASE-3 aggregation inference COMPLETE")
print(f"🎧 Total audio files evaluated: {len(audio_results)}")
