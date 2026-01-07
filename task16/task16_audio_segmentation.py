import os
import librosa
import soundfile as sf

# =================================================
# CONFIGURATION
# =================================================
DATASET_DIR = "IRMAS-TrainingData/IRMAS-TrainingData"

OUTPUT_BASE = "task16/outputs"
SEGMENTS_DIR = os.path.join(OUTPUT_BASE, "segments")

SEGMENT_DURATION = 3.0   # seconds (IRMAS clips ~3 sec)
OVERLAP = 1.5
SR = 22050

# Create required directories
os.makedirs(SEGMENTS_DIR, exist_ok=True)

# IRMAS instrument mapping
instrument_folders = {
    "guitar": ["gac", "gel"],
    "piano": ["pia"],
    "flute": ["flu"]
}

# =================================================
# FUNCTION: SEGMENT AUDIO
# =================================================
def segment_audio(file_path, instrument):
    y, sr = librosa.load(file_path, sr=SR)

    segment_len = int(SEGMENT_DURATION * sr)

    instrument_dir = os.path.join(SEGMENTS_DIR, instrument)
    os.makedirs(instrument_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # IRMAS clips are ~3 seconds → keep full clip as one segment
    if len(y) <= segment_len:
        out_file = os.path.join(
            instrument_dir,
            f"{base_name}_seg0.wav"
        )
        sf.write(out_file, y, sr)
        return 1

    # For longer audio (future-proof)
    hop_len = int((SEGMENT_DURATION - OVERLAP) * sr)
    count = 0

    for start in range(0, len(y) - segment_len + 1, hop_len):
        segment = y[start:start + segment_len]
        out_file = os.path.join(
            instrument_dir,
            f"{base_name}_seg{count}.wav"
        )
        sf.write(out_file, segment, sr)
        count += 1

    return count

# =================================================
# MAIN PROCESS
# =================================================
total_segments = 0
processed_files = 0

for instrument, folders in instrument_folders.items():
    for folder in folders:
        folder_path = os.path.join(DATASET_DIR, folder)

        if not os.path.exists(folder_path):
            continue

        files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(".wav")
        ]

        for f in files[:3]:  # limit for demo
            file_path = os.path.join(folder_path, f)
            total_segments += segment_audio(file_path, instrument)
            processed_files += 1

print("======================================")
print("✅ Audio segmentation completed")
print(f"Files processed     : {processed_files}")
print(f"Segments generated  : {total_segments}")
print("======================================")
