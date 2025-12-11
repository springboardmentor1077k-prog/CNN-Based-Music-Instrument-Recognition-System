import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
SOURCE_PATH = './IRMAS-TrainingData/IRMAS-TrainingData'
OUTPUT_FOLDER = './Augmented_Samples'
TARGET_SR = 16000

# We pick 3 distinct instruments for the demonstration
SELECTED_INSTRUMENTS = ['pia', 'sax', 'cel'] 
# pia = Piano, sax = Saxophone, cel = Cello

def add_white_noise(data, noise_factor=0.005):
    """
    Adds random static noise to the audio.
    noise_factor: Controls how loud the noise is (0.005 is subtle, 0.02 is loud)
    """
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to float32 to ensure compatibility
    return augmented_data.astype(type(data[0]))

def time_shift(data, sr, shift_seconds=0.5):
    """
    Shifts the audio forward/backward in time.
    We roll the array.
    """
    shift_samples = int(shift_seconds * sr)
    # np.roll moves the data. Data from the end wraps to the start.
    return np.roll(data, shift_samples)

def change_pitch(data, sr, steps=2):
    """
    Changes the pitch without changing the speed.
    steps: Number of semitones (2 = 1 whole tone up, -2 = 1 whole tone down)
    """
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=steps)

def run_augmentations():
    # 1. Setup Output Folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    if not os.path.exists(SOURCE_PATH):
        print(f"❌ Error: Source folder {SOURCE_PATH} not found.")
        return

    plt.figure(figsize=(15, 10))
    plot_idx = 1

    # 2. Loop through the 3 selected instruments
    for instr in SELECTED_INSTRUMENTS:
        folder_path = os.path.join(SOURCE_PATH, instr)
        
        # Get the first .wav file found
        files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        if not files:
            print(f"Skipping {instr}: No files found.")
            continue
            
        first_file = files[0] # Prefer first audio as requested
        file_path = os.path.join(folder_path, first_file)
        print(f"\n🎵 Processing: {instr} / {first_file}")

        # Load Audio
        y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
        
        # --- APPLY AUGMENTATIONS ---
        
        # A. Pitch Shift (Up 2 semitones)
        y_pitch = change_pitch(y, sr, steps=2)
        
        # B. Add Background Noise
        y_noise = add_white_noise(y, noise_factor=0.01)
        
        # C. Time Shift (Shift right by 0.5 seconds)
        y_shift = time_shift(y, sr, shift_seconds=0.5)

        # --- SAVE FILES ---
        base_name = first_file.replace('.wav', '')
        sf.write(os.path.join(OUTPUT_FOLDER, f"{base_name}_original.wav"), y, sr)
        sf.write(os.path.join(OUTPUT_FOLDER, f"{base_name}_pitch_up.wav"), y_pitch, sr)
        sf.write(os.path.join(OUTPUT_FOLDER, f"{base_name}_noise.wav"), y_noise, sr)
        sf.write(os.path.join(OUTPUT_FOLDER, f"{base_name}_shifted.wav"), y_shift, sr)

        # --- PLOT WAVEFORMS ---
        # Original
        plt.subplot(3, 4, plot_idx)
        librosa.display.waveshow(y, sr=sr, alpha=0.6)
        plt.title(f"{instr} - Original")
        plot_idx += 1
        
        # Pitch Shift
        plt.subplot(3, 4, plot_idx)
        librosa.display.waveshow(y_pitch, sr=sr, color='g', alpha=0.6)
        plt.title(f"{instr} - Pitch Shift (+2)")
        plot_idx += 1

        # Noise
        plt.subplot(3, 4, plot_idx)
        librosa.display.waveshow(y_noise, sr=sr, color='r', alpha=0.6)
        plt.title(f"{instr} - Added Noise")
        plot_idx += 1

        # Time Shift
        plt.subplot(3, 4, plot_idx)
        librosa.display.waveshow(y_shift, sr=sr, color='purple', alpha=0.6)
        plt.title(f"{instr} - Time Shift")
        plot_idx += 1

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "augmentation_comparison.png"))
    print(f"\n✅ Done! Check the folder: {OUTPUT_FOLDER}")
    plt.show()

if __name__ == "__main__":
    run_augmentations()