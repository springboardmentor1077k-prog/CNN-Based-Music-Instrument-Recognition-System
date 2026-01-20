import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_DIR = r"C:\Users\ADMIN\Downloads\music_dataset"
OUT_DIR = "cnn_new/data/audio_csv"
os.makedirs(OUT_DIR, exist_ok=True)

rows = []

for label in sorted(os.listdir(DATASET_DIR)):
    class_dir = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(class_dir):
        continue

    for f in os.listdir(class_dir):
        if f.endswith(".wav"):
            rows.append({
                "path": os.path.join(class_dir, f),
                "label": label
            })

df = pd.DataFrame(rows)
print("Total files:", len(df))
print(df["label"].value_counts())

train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df["label"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
)

train_df.to_csv(f"{OUT_DIR}/train.csv", index=False)
val_df.to_csv(f"{OUT_DIR}/val.csv", index=False)
test_df.to_csv(f"{OUT_DIR}/test.csv", index=False)

print("✅ CSVs created")
