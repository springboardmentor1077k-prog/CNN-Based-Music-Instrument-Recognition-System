import os
import glob
import json

import librosa
import pandas as pd

# Base folder where your IRMAS instrument folders are stored
# Change this if your path is slightly different
IRMAS_BASE = "IRMAS-TrainingData/IRMAS-TrainingData"

# Map logical instrument name -> IRMAS folder code
INSTRUMENT_DIRS = {
    "guitar": "gac",  # acoustic guitar
    "piano": "pia",
    "flute": "flu",
}

# Map instrument -> family
INSTRUMENT_FAMILY = {
    "guitar": "strings",
    "piano": "keyboard",
    "flute": "woodwind",
}

SOURCE_NAME = "IRMAS Dataset"

def pick_example_files():
    """Pick one example .wav file for each requested instrument."""
    file_paths = {}
    for inst, folder in INSTRUMENT_DIRS.items():
        pattern = os.path.join(IRMAS_BASE, folder, "*.wav")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No .wav files found for {inst} in {pattern}")
        file_paths[inst] = files[0]  # take the first file
    return file_paths

def inspect_metadata(file_paths):
    """Load each audio file and print basic metadata."""
    print("=== Audio metadata inspection ===")
    metadata_entries = []

    for inst, path in file_paths.items():
        y, sr = librosa.load(path, sr=None)
        duration_sec = len(y) / sr
        num_samples = len(y)

        print(f"\nInstrument: {inst}")
        print(f"  File: {path}")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Duration: {duration_sec:.2f} seconds")
        print(f"  Number of samples: {num_samples}")

        metadata_entries.append(
            {
                "filename": os.path.basename(path),
                "full_path": path,
                "instrument": inst,
                "family": INSTRUMENT_FAMILY[inst],
                "sample_rate": sr,
                "duration_sec": round(duration_sec, 2),
                "num_samples": num_samples,
                "source": SOURCE_NAME,
            }
        )

    return metadata_entries

def create_json_labels(metadata_entries, json_path="day3_instrument_labels.json"):
    """Save metadata/labels to a JSON file."""
    data = {
        entry["filename"]: {
            "instrument": entry["instrument"],
            "family": entry["family"],
            "source": entry["source"],
        }
        for entry in metadata_entries
    }

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"\nSaved JSON labels to {json_path}")
    return json_path

def print_labels_from_json(json_path):
    """Load the JSON file and print the instrument labels."""
    print("\n=== Instrument labels from JSON ===")
    with open(json_path, "r") as f:
        data = json.load(f)

    for filename, info in data.items():
        print(f"{filename}: instrument={info['instrument']}, family={info['family']}, source={info['source']}")

def create_mapping_table(metadata_entries, csv_path="day3_instrument_mapping.csv"):
    """Create a mapping table: filename, instrument, family, source."""
    df = pd.DataFrame(
        [
            {
                "filename": entry["filename"],
                "instrument": entry["instrument"],
                "family": entry["family"],
                "source": entry["source"],
            }
            for entry in metadata_entries
        ]
    )

    print("\n=== Instrument mapping table ===")
    print(df)

    df.to_csv(csv_path, index=False)
    print(f"\nSaved mapping table to {csv_path}")
    return csv_path

if __name__ == "__main__":
    # 1. Pick example files for guitar, piano, flute
    example_files = pick_example_files()

    # 2. Inspect metadata (sr, duration, num samples)
    metadata_entries = inspect_metadata(example_files)

    # 3. Create JSON with labels and print labels from JSON
    json_path = create_json_labels(metadata_entries)
    print_labels_from_json(json_path)

    # 4. Create mapping table (filename, instrument, family, source)
    create_mapping_table(metadata_entries)
