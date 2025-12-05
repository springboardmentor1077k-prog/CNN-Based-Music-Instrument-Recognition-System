import json
import pandas as pd
import os
import librosa
import numpy as np

def process_nsynth_data(nsynth_json_path, nsynth_audio_dir, output_csv_path, num_samples_to_load=3):
    """
    Processes the NSynth dataset to create a mapping table and load sample audio files.

    Args:
        nsynth_json_path (str): Path to the examples.json file.
        nsynth_audio_dir (str): Path to the directory containing NSynth audio files.
        output_csv_path (str): Path to save the generated mapping table CSV.
        num_samples_to_load (int): Number of audio samples to load for demonstration.
    """
    print(f"Loading NSynth metadata from: {nsynth_json_path}")
    with open(nsynth_json_path, 'r') as f:
        nsynth_metadata = json.load(f)

    mapping_data = []
    sample_audio_files = []

    print("Extracting metadata and preparing mapping table...")
    for note_id, data in nsynth_metadata.items():
        instrument_str = data.get('instrument_str', 'unknown')
        instrument_family_str = data.get('instrument_family_str', 'unknown')
        
        # NSynth audio files are named after the note_id with a .wav extension
        wav_filename = f"{note_id}.wav"
        full_audio_path = os.path.join(nsynth_audio_dir, wav_filename)

        mapping_data.append({
            'file_name': wav_filename,
            'instrument': instrument_str,
            'family': instrument_family_str,
            'source': 'NSynth',
            'full_path': full_audio_path
        })
        
        if len(sample_audio_files) < num_samples_to_load and os.path.exists(full_audio_path):
            sample_audio_files.append(full_audio_path)

    nsynth_df = pd.DataFrame(mapping_data)
    print(f"Generated mapping table with {len(nsynth_df)} entries.")
    
    # Save the mapping table
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    nsynth_df.to_csv(output_csv_path, index=False)
    print(f"Mapping table saved to: {output_csv_path}")

    # Load and print info for sample audio files
    loaded_samples = []
    print(f"\nLoading {len(sample_audio_files)} sample audio files:")
    for i, audio_path in enumerate(sample_audio_files):
        try:
            y, sr = librosa.load(audio_path, sr=None)
            print(f"  Sample {i+1}: {os.path.basename(audio_path)} - Shape: {y.shape}, Sample Rate: {sr} Hz, Duration: {len(y)/sr:.2f}s")
            loaded_samples.append((y, sr, audio_path))
        except Exception as e:
            print(f"  Error loading {os.path.basename(audio_path)}: {e}")
            
    return nsynth_df, loaded_samples

if __name__ == '__main__':
    NSYNTH_BASE_DIR = 'datasets/nsynth-training-data/nsynth-train'
    NSYNTH_JSON_PATH = os.path.join(NSYNTH_BASE_DIR, 'examples.json')
    NSYNTH_AUDIO_DIR = os.path.join(NSYNTH_BASE_DIR, 'audio')
    OUTPUT_CSV_PATH = 'outputs/nsynth_mapping_table.csv'

    # Ensure librosa is installed
    try:
        import librosa
    except ImportError:
        print("librosa not found. Please install it: pip install librosa")
        exit()

    processed_df, loaded_audio = process_nsynth_data(
        nsynth_json_path=NSYNTH_JSON_PATH,
        nsynth_audio_dir=NSYNTH_AUDIO_DIR,
        output_csv_path=OUTPUT_CSV_PATH,
        num_samples_to_load=3
    )

    print("\nProcessing complete.")
