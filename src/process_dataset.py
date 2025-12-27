import os
import sys
import soundfile as sf
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audio_processor import process_audio_file, add_noise, pitch_shift, adjust_volume, time_shift
from visualizer import save_clean_spectrogram

def main():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "IRMAS-TrainingData")
    OUTPUT_WAV_DIR = os.path.join(PROJECT_ROOT, "outputs", "processed_irmas")
    OUTPUT_SPEC_DIR = os.path.join(PROJECT_ROOT, "outputs", "mel_spectrograms_irmas")
    
    print(f"Starting processing pipeline...")
    print(f"Input Directory: {DATASET_DIR}")
    print(f"Audio Output Directory: {OUTPUT_WAV_DIR}")
    print(f"Spectrogram Output Directory: {OUTPUT_SPEC_DIR}")

    if not os.path.exists(DATASET_DIR):
        print("Dataset directory not found!")
        return

    # Create root output directories first to avoid potential race conditions or path errors
    os.makedirs(OUTPUT_WAV_DIR, exist_ok=True)
    os.makedirs(OUTPUT_SPEC_DIR, exist_ok=True)

    # Count total files for progress bar
    total_files = sum([len(files) for r, d, files in os.walk(DATASET_DIR) if any(f.endswith('.wav') for f in files)])
    
    processed_count = 0
    failed_count = 0

    with tqdm(total=total_files, desc="Processing Audio & Generating Spectrograms") as pbar:
        for root, dirs, files in os.walk(DATASET_DIR):
            # Get class name from folder name
            if root == DATASET_DIR:
                continue
            
            class_name = os.path.basename(root)

            # Skip hidden directories (like .ipynb_checkpoints) or system folders
            if class_name.startswith('.') or class_name.startswith('_'):
                continue
            
            # Create class subdirectories
            wav_class_dir = os.path.join(OUTPUT_WAV_DIR, class_name)
            spec_class_dir = os.path.join(OUTPUT_SPEC_DIR, class_name)
            
            os.makedirs(wav_class_dir, exist_ok=True)
            os.makedirs(spec_class_dir, exist_ok=True)

            for file in files:
                if file.endswith('.wav'):
                    input_path = os.path.join(root, file)
                    
                    # Base filename without extension
                    base_name = os.path.splitext(file)[0]

                    # Process Audio (Base Version)
                    y, sr = process_audio_file(input_path, target_sr=16000, duration=3.0)

                    if y is not None:
                        # Define variations: (suffix, audio_data)
                        variations = [
                            ("", y),  # Original
                            ("_noise", add_noise(y, noise_level=0.01)),  # Noise Injection
                            ("_pitch", pitch_shift(y, sr, n_steps=2)),   # Pitch Shift (+2 semitones)
                            ("_vol", adjust_volume(y, factor=0.8)),      # Volume Adjustment (0.8x)
                            ("_shift", time_shift(y, sr, shift_max=0.5)) # Time Shift (Rolling)
                        ]
                        
                        for suffix, y_var in variations:
                            # Construct paths
                            wav_name = f"{base_name}{suffix}.wav"
                            png_name = f"{base_name}{suffix}.png"
                            
                            wav_path = os.path.join(wav_class_dir, wav_name)
                            spec_path = os.path.join(spec_class_dir, png_name)
                            
                            # 1. Save Audio
                            sf.write(wav_path, y_var, sr)
                            
                            # 2. Save Spectrogram
                            save_clean_spectrogram(y_var, sr, spec_path)

                        processed_count += 1
                    else:
                        failed_count += 1
                    
                    pbar.update(1)

    print(f"\nProcessing Complete.")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed: {failed_count}")
    print(f"Audio saved to: {OUTPUT_WAV_DIR}")
    print(f"Spectrograms saved to: {OUTPUT_SPEC_DIR}")

if __name__ == "__main__":
    main()
