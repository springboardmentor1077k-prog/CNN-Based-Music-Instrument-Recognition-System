from datasets import load_dataset, Audio
import soundfile as sf
import numpy as np
import os
import io

# ========= CREATE TASK-3 OUTPUT FOLDER ==========
task3_dir = "task3"
os.makedirs(task3_dir, exist_ok=True)

audio_dir = os.path.join(task3_dir, "audio")
os.makedirs(audio_dir, exist_ok=True)

# ========= 1. LOAD DATASET WITHOUT AUDIO DECODING ==========
ds = load_dataset("mteb/nsynth-mini")
ds = ds.cast_column("audio", Audio(decode=False))

train = ds["train"]

print("Columns in the dataset:")
print(train.column_names)

# ========= 2. INSPECT 3 METADATA ENTRIES ==========
metadata_path = os.path.join(task3_dir, "metadata.txt")
meta_file = open(metadata_path, "w", encoding="utf-8")

print("\n=== Inspect 3 metadata entries (train split) ===\n")

examples = []

for i in range(3):
    ex = train[i]
    examples.append(ex)

    meta_file.write(f"Example {i}:\n")
    meta_file.write(f"  note: {ex['note']}\n")
    meta_file.write(f"  pitch: {ex['pitch']}\n")
    meta_file.write(f"  velocity: {ex['velocity']}\n")
    meta_file.write(f"  instrument id: {ex['instrument']}\n")
    meta_file.write(f"  instrument name: {ex['instrument_str']}\n")
    meta_file.write(f"  instrument family id: {ex['instrument_family']}\n")
    meta_file.write(f"  instrument family name: {ex['instrument_family_str']}\n")
    meta_file.write(f"  instrument source: {ex['instrument_source_str']}\n")
    meta_file.write(f"  qualities: {ex['qualities_str']}\n")
    meta_file.write("-" * 50 + "\n")

meta_file.close()
print(f"Metadata saved to: {metadata_path}")

# ========= 3. SAVE 3 AUDIO FILES ==========
print("\n=== Saving audio files to task3/audio/ ===\n")

for i, ex in enumerate(examples):
    audio_field = ex["audio"]

    if isinstance(audio_field, dict) and "bytes" in audio_field:
        audio_bytes = audio_field["bytes"]
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        audio = np.array(data, dtype="float32")
    else:
        raise TypeError("Unexpected audio format")

    fname = f"example_{i}_{ex['instrument_family_str']}_{ex['instrument_str']}.wav"
    path = os.path.join(audio_dir, fname)

    sf.write(path, audio, sr)
    print(f"Saved: {path}")

# ========= 4. SAVE INSTRUMENT MAPPING TABLE ==========
mapping_path = os.path.join(task3_dir, "instrument_mapping.txt")

with open(mapping_path, "w", encoding="utf-8") as f:
    f.write("Example | Inst_ID | Instrument | Fam_ID | Family\n")
    f.write("-" * 70 + "\n")

    for i, ex in enumerate(examples):
        f.write(
            f"{i:<7} | "
            f"{ex['instrument']:<7} | "
            f"{ex['instrument_str']:<25} | "
            f"{ex['instrument_family']:<6} | "
            f"{ex['instrument_family_str']}\n"
        )

print(f"\nInstrument mapping saved to: {mapping_path}")

# ========= DONE ==========
print("\nTask 3 completed successfully. Files saved inside 'task3/' folder.")
