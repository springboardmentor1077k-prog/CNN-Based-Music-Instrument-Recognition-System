import os
import sys
import soundfile as sf
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audio_processor import process_audio_file
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
                    
                    # Output paths
                    wav_output_path = os.path.join(wav_class_dir, file)
                    spec_output_path = os.path.join(spec_class_dir, file.replace('.wav', '.png'))

                    # Process Audio
                    y, sr = process_audio_file(input_path, target_sr=16000, duration=3.0)

                    if y is not None:
                        # 1. Save processed audio
                        sf.write(wav_output_path, y, sr)
                        
                        # 2. Generate and Save Mel Spectrogram (Clean version for CNN)
                        save_clean_spectrogram(
                            y, 
                            sr, 
                            spec_output_path
                        )
                        
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
