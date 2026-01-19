import os
import sys
import argparse

# Fix for Numba/Librosa permission issue in Docker (non-root)
os.environ['NUMBA_CACHE_DIR'] = '/tmp'

import soundfile as sf
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audio_preprocessor import process_audio_file
from augmentation import add_noise, pitch_shift, adjust_volume, time_shift
from visualizer import save_clean_spectrogram, save_hpss_image

def main():
    parser = argparse.ArgumentParser(description="Preprocess IRMAS dataset with Split-First strategy")
    parser.add_argument("--input_dir", type=str, default=None, help="Path to raw IRMAS-TrainingData")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to save processed data")
    args = parser.parse_args()

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Resolve Paths
    if args.input_dir:
        DATASET_DIR = args.input_dir
    else:
        DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "IRMAS-TrainingData")
        
    if args.output_dir:
        PROCESSED_ROOT = args.output_dir
    else:
        PROCESSED_ROOT = os.path.join(PROJECT_ROOT, "datasets", "IRMAS-ProcessedTrainingData")
    
    # Define Train/Val Split Directories
    TRAIN_AUDIO_DIR = os.path.join(PROCESSED_ROOT, "train", "audio")
    TRAIN_SPEC_DIR = os.path.join(PROCESSED_ROOT, "train", "spectrograms")
    VAL_AUDIO_DIR = os.path.join(PROCESSED_ROOT, "validation", "audio")
    VAL_SPEC_DIR = os.path.join(PROCESSED_ROOT, "validation", "spectrograms")
    
    print(f"Starting processing pipeline with Split-First strategy...")
    print(f"Input Directory: {DATASET_DIR}")
    print(f"Train Output: {TRAIN_SPEC_DIR}")
    print(f"Validation Output: {VAL_SPEC_DIR}")

    if not os.path.exists(DATASET_DIR):
        print("Dataset directory not found!")
        return

    # Create root output directories
    for d in [TRAIN_AUDIO_DIR, TRAIN_SPEC_DIR, VAL_AUDIO_DIR, VAL_SPEC_DIR]:
        os.makedirs(d, exist_ok=True)

    # Get list of classes
    classes = [d for d in os.listdir(DATASET_DIR) 
               if os.path.isdir(os.path.join(DATASET_DIR, d)) 
               and not d.startswith('.') and not d.startswith('_')]

    processed_count = 0
    failed_count = 0

    print(f"Found classes: {classes}")

    for class_name in tqdm(classes, desc="Processing Classes"):
        class_dir = os.path.join(DATASET_DIR, class_name)
        files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
        
        # Split files into Train and Validation
        train_files, val_files = train_test_split(files, test_size=0.2, random_state=42, shuffle=True)
        
        # Helper function to process files
        def process_batch(file_list, is_train):
            nonlocal processed_count, failed_count
            
            # Determine output directories based on split
            if is_train:
                out_audio_dir = os.path.join(TRAIN_AUDIO_DIR, class_name)
                out_spec_dir = os.path.join(TRAIN_SPEC_DIR, class_name)
            else:
                out_audio_dir = os.path.join(VAL_AUDIO_DIR, class_name)
                out_spec_dir = os.path.join(VAL_SPEC_DIR, class_name)
            
            os.makedirs(out_audio_dir, exist_ok=True)
            os.makedirs(out_spec_dir, exist_ok=True)

            for file in file_list:
                input_path = os.path.join(class_dir, file)
                base_name = os.path.splitext(file)[0]
                
                # Process Audio (Base Version)
                y, sr = process_audio_file(input_path, target_sr=16000, duration=3.0)

                if y is not None:
                    # Define variations
                    if is_train:
                        # Train: Apply Augmentations
                        variations = [
                            ("", y),  # Original
                            ("_noise", add_noise(y, noise_level=0.01)),
                            ("_pitch", pitch_shift(y, sr, n_steps=2)),
                            ("_vol", adjust_volume(y, factor=0.8)),
                            ("_shift", time_shift(y, sr, shift_max=0.5))
                        ]
                    else:
                        # Validation: Original ONLY
                        variations = [("", y)]
                    
                    for suffix, y_var in variations:
                        wav_name = f"{base_name}{suffix}.wav"
                        png_name = f"{base_name}{suffix}.png"
                        
                        wav_path = os.path.join(out_audio_dir, wav_name)
                        spec_path = os.path.join(out_spec_dir, png_name)
                        
                        # Save Audio & HPSS Spectrogram
                        sf.write(wav_path, y_var, sr)
                        save_hpss_image(y_var, sr, spec_path)

                    processed_count += 1
                else:
                    failed_count += 1

        # Process batches
        process_batch(train_files, is_train=True)
        process_batch(val_files, is_train=False)

    print(f"\nProcessing Complete.")
    print(f"Original Files Processed: {processed_count}")
    print(f"Failed: {failed_count}")
    print(f"Data saved to: {PROCESSED_ROOT}")

if __name__ == "__main__":
    main()
