import os
import random
import soundfile as sf
import numpy as np
import audio_processor as ap
import matplotlib.pyplot as plt
import librosa
import librosa.display
from glob import glob

def plot_comparison(audio_data, titles, sr, output_path):
    """
    Plots Mel Spectrograms for a list of audio signals in a single figure.
    """
    n = len(audio_data)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=False)
    
    if n == 1:
        axes = [axes]
        
    for i, (y, title) in enumerate(zip(audio_data, titles)):
        ax = axes[i]
        M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, power=1.0)
        M_db = librosa.amplitude_to_db(M, ref=np.max)
        img = librosa.display.specshow(M_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        ax.set_title(title)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  Saved Comparison Plot: {output_path}")

def run_augmentation_demo():
    # Define paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "IRMAS-TrainingData")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "augmented_samples")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find all wav files
    all_wavs = []
    for root, _, files in os.walk(DATASET_DIR):
        for f in files:
            if f.lower().endswith('.wav'):
                all_wavs.append(os.path.join(root, f))
                
    if len(all_wavs) < 3:
        print("Not enough audio files found.")
        return

    # Select 3 random files
    selected_files = random.sample(all_wavs, 3)
    
    print(f"Selected {len(selected_files)} files for augmentation demonstration.")
    
    for i, file_path in enumerate(selected_files):
        filename = os.path.basename(file_path)
        print(f"\nProcessing [{i+1}/3]: {filename}")
        
        # Load and preprocess (Baseline)
        # using process_audio_file to get consistent 16kHz mono
        y, sr = ap.process_audio_file(file_path)
        
        if y is None:
            print("Skipping due to load error.")
            continue
            
        base_name = os.path.splitext(filename)[0]
        
        # Save Original (Processed)
        orig_path = os.path.join(OUTPUT_DIR, f"{base_name}_original.wav")
        sf.write(orig_path, y, sr)
        print(f"  Saved: {orig_path}")
        
        # Store for plotting
        audio_list = [y]
        title_list = ["Original"]
        
        # 1. Add Noise
        y_noise = ap.add_noise(y, noise_level=0.02)
        noise_path = os.path.join(OUTPUT_DIR, f"{base_name}_noise.wav")
        sf.write(noise_path, y_noise, sr)
        print(f"  Saved (Noise): {noise_path}")
        audio_list.append(y_noise)
        title_list.append("Add Noise (0.02)")
        
        # 2. Time Stretch (Speed up by 1.2x)
        # Note: Time stretch changes duration.
        y_stretch = ap.time_stretch(y, rate=1.2)
        stretch_path = os.path.join(OUTPUT_DIR, f"{base_name}_stretch.wav")
        sf.write(stretch_path, y_stretch, sr)
        print(f"  Saved (Time Stretch 1.2x): {stretch_path}")
        audio_list.append(y_stretch)
        title_list.append("Time Stretch (1.2x)")
        
        # 3. Pitch Shift (Up by 4 semitones)
        y_shift = ap.pitch_shift(y, sr, n_steps=4)
        shift_path = os.path.join(OUTPUT_DIR, f"{base_name}_pitch_shift.wav")
        sf.write(shift_path, y_shift, sr)
        print(f"  Saved (Pitch Shift +4): {shift_path}")
        audio_list.append(y_shift)
        title_list.append("Pitch Shift (+4 semitones)")

        # Plot Comparison
        plot_path = os.path.join(OUTPUT_DIR, f"{base_name}_comparison.png")
        plot_comparison(audio_list, title_list, sr, plot_path)

    print(f"\nAll operations completed. Check {OUTPUT_DIR} for results.")

if __name__ == "__main__":
    run_augmentation_demo()
