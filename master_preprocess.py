import librosa
import librosa.display
import librosa.effects
import librosa.util
import soundfile as sf  # Required to save new audio files
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. CONFIGURATION ---
SOURCE_PATH = './IRMAS-TrainingData/IRMAS-TrainingData'
OUTPUT_AUDIO_PATH = './Cleaned_Audio_Dataset'
OUTPUT_IMAGE_PATH = './Mel_Spectrogram_Dataset'

TARGET_SR = 16000          # Step 3: 16kHz
FIXED_DURATION = 3.0       # Step 6: 3 Seconds per file
target_samples = int(FIXED_DURATION * TARGET_SR) # 3 * 16000 = 48,000 samples

def master_pipeline():
    # Setup Output Directories
    for path in [OUTPUT_AUDIO_PATH, OUTPUT_IMAGE_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)

    # Get Instrument Folders
    if not os.path.exists(SOURCE_PATH):
        print("Error: Source folder not found!")
        return
    
    instruments = [d for d in os.listdir(SOURCE_PATH) if os.path.isdir(os.path.join(SOURCE_PATH, d))]
    print(f"Found Instruments: {instruments}")

    for instrument in instruments:
        print(f"\n--- Processing Instrument: {instrument} ---")
        
        # Define paths for this specific instrument
        src_folder = os.path.join(SOURCE_PATH, instrument)
        dest_audio_folder = os.path.join(OUTPUT_AUDIO_PATH, instrument)
        dest_image_folder = os.path.join(OUTPUT_IMAGE_PATH, instrument)
        
        # Create subfolders
        os.makedirs(dest_audio_folder, exist_ok=True)
        os.makedirs(dest_image_folder, exist_ok=True)
        
        files = [f for f in os.listdir(src_folder) if f.endswith('.wav')]
        
        for i, audio_file in enumerate(files):
            try:
                file_path = os.path.join(src_folder, audio_file)
                
                # --- STEP 1 & 2 & 3: Load, Mono, Resample ---
                # This loads the audio as a floating point number
                y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
                
                # --- STEP 4: Audio Normalization ---
                # Normalize amplitude to range [-1, 1] so volume is consistent
                y = librosa.util.normalize(y)
                
                # --- STEP 5: Silence Trimming ---
                # Removes leading and trailing silence (below 60db)
                y_trimmed, _ = librosa.effects.trim(y, top_db=60)
                
                # --- STEP 6: Fix Duration (Pad or Truncate) ---
                # If audio is < 3s, pad with zeros. If > 3s, cut it.
                if len(y_trimmed) < target_samples:
                    # Pad (add silence to the end)
                    y_fixed = librosa.util.fix_length(y_trimmed, size=target_samples)
                else:
                    # Truncate (cut the extra off)
                    y_fixed = y_trimmed[:target_samples]
                
                # --- EXPORT 1: Save Cleaned Audio ---
                save_audio_path = os.path.join(dest_audio_folder, audio_file)
                sf.write(save_audio_path, y_fixed, TARGET_SR)
                
                # --- EXPORT 2: Generate & Save Mel-Spectrogram ---
                spectrogram = librosa.feature.melspectrogram(y=y_fixed, sr=TARGET_SR, n_mels=128)
                spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
                
                plt.figure(figsize=(4, 4))
                librosa.display.specshow(spectrogram_db, sr=TARGET_SR)
                plt.axis('off') # No borders/text
                
                save_image_name = audio_file.replace('.wav', '.png')
                save_image_path = os.path.join(dest_image_folder, save_image_name)
                
                plt.savefig(save_image_path, bbox_inches='tight', pad_inches=0)
                plt.close() # Important: free memory
                
                if i % 50 == 0:
                    print(f"Processed {i}/{len(files)} files")
                    
            except Exception as e:
                print(f"Error on {audio_file}: {e}")

    print("\n✅ PREPROCESSING COMPLETE")
    print(f"1. Clean Audio saved to: {OUTPUT_AUDIO_PATH}")
    print(f"2. Spectrograms saved to: {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    master_pipeline()