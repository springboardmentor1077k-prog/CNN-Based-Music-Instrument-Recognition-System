import os
import shutil
import numpy as np

# SOURCE = where extract_features.py saved data
SRC_DIR = r"C:\Users\ADMIN\Downloads\check"

# DESTINATION = cnn_new workspace
BASE_DIR = r"C:\Users\ADMIN\Downloads\check\cnn_new"
FEATURE_DIR = os.path.join(BASE_DIR, "features")
os.makedirs(FEATURE_DIR, exist_ok=True)

FILES = [
    "X_train.npy", "y_train.npy",
    "X_val.npy",   "y_val.npy",
    "X_test.npy",  "y_test.npy"
]

print("🔍 Moving feature files into cnn_new/features")

for f in FILES:
    src = os.path.join(SRC_DIR, f)
    dst = os.path.join(FEATURE_DIR, f)

    if not os.path.exists(src):
        raise FileNotFoundError(f"❌ Missing source file: {src}")

    data = np.load(src)
    np.save(dst, data)
    print(f"✅ {f} → features/  shape={data.shape}")

print("\n🎉 Phase-1 features correctly prepared for Phase-2")
