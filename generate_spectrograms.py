import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
SOURCE_PATH = './IRMAS-TrainingData/IRMAS-TrainingData' # Your current audio folder
DESTINATION_PATH = './Processed_Spectrograms'           # Where images will be saved
TARGET_SR = 16000                                       # 16kHz

def create_spectrograms():
    # 1. Create the main destination folder if it doesn't exist
    if not os.path.exists(DESTINATION_PATH):
        os.makedirs(DESTINATION_PATH)
        print(f"Created folder: {DESTINATION_PATH}")

    # 2. Get list of instrument folders
    if not os.path.exists(SOURCE_PATH):
        print(f"Error: Source path '{SOURCE_PATH}' not found!")
        return
        
    instruments = [d for d in os.listdir(SOURCE_PATH) if os.path.isdir(os.path.join(SOURCE_PATH, d))]
    print(f"Found instruments: {instruments}")

    # 3. Loop through EACH instrument folder
    for instrument in instruments:
        source_folder = os.path.join(SOURCE_PATH, instrument)
        dest_folder = os.path.join(DESTINATION_PATH, instrument)
        
        # Create a matching folder in destination (e.g., Processed_Spectrograms/pia)
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
            
        print(f"Processing {instrument}...")
        
        files = [f for f in os.listdir(source_folder) if f.endswith('.wav')]
        
        count = 0
        for audio_file in files:
            try:
                # A. Load Audio (Mono, 16kHz)
                file_path = os.path.join(source_folder, audio_file)
                y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
                
                # B. Generate Mel-Spectrogram
                # n_mels=128 creates an image with 128 pixels height
                spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
                
                # C. Save as Image (removing axes/labels for pure data)
                plt.figure(figsize=(4, 4))
                librosa.display.specshow(spectrogram_db, sr=sr)
                plt.axis('off') # Remove x/y axis numbers
                
                # Save filename: "original_name.png"
                save_name = audio_file.replace('.wav', '.png')
                save_path = os.path.join(dest_folder, save_name)
                
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close() # Close plot to free memory
                
                count += 1
                if count % 50 == 0:
                    print(f"  - Converted {count} files in {instrument}")
                    
            except Exception as e:
                print(f"  ! Error processing {audio_file}: {e}")

    print("\n✅ All Finished! You now have an IMAGE dataset ready for the CNN.")

if __name__ == "__main__":
    create_spectrograms()