import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from audio_preprocessor import process_audio_file

# Define paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
FILE_PATHS = {
    "Cello (cel)": "datasets/IRMAS-TrainingData/cel/008__[cel][nod][cla]0058__1.wav",
    "Clarinet (cla)": "datasets/IRMAS-TrainingData/cla/004__[cla][nod][cla]0242__1.wav",
    "Flute (flu)": "datasets/IRMAS-TrainingData/flu/008__[flu][nod][cla]0393__1.wav",
    "Acoustic Guitar (gac)": "datasets/IRMAS-TrainingData/gac/014__[gac][nod][cou_fol]0770__1.wav",
    "Electric Guitar (gel)": "datasets/IRMAS-TrainingData/gel/001__[gel][dru][pop_roc]0829__1.wav",
    "Organ (org)": "datasets/IRMAS-TrainingData/org/001__[org][dru][jaz_blu]1123__1.wav",
    "Piano (pia)": "datasets/IRMAS-TrainingData/pia/001__[pia][nod][cla]1389__1.wav",
    "Saxophone (sax)": "datasets/IRMAS-TrainingData/sax/006__[sax][nod][cla]1686__1.wav",
    "Trumpet (tru)": "datasets/IRMAS-TrainingData/tru/001__[tru][nod][jaz_blu]1986__1.wav",
    "Violin (vio)": "datasets/IRMAS-TrainingData/vio/001__[vio][nod][cou_fol]2194__1.wav",
    "Voice (voi)": "datasets/IRMAS-TrainingData/voi/001__[voi][dru][pop_roc]2321__1.wav",
}

def compare_instruments():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Starting instrument comparison visualization...")
    
    # 11 instruments, 2 columns (STFT, Mel)
    fig, axes = plt.subplots(nrows=11, ncols=2, figsize=(15, 33))
    # Adjust spacing
    plt.subplots_adjust(hspace=0.6, wspace=0.3)

    for i, (instrument, rel_path) in enumerate(FILE_PATHS.items()):
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        print(f"Processing {instrument}...")
        
        # Load and process
        y, sr = process_audio_file(full_path)
        
        if y is None:
            print(f"Failed to process {instrument}")
            continue

        # --- STFT ---
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        ax_stft = axes[i, 0]
        img_stft = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=ax_stft)
        ax_stft.set_title(f"{instrument} - STFT")
        fig.colorbar(img_stft, ax=ax_stft, format='%+2.0f dB')

        # --- Mel Spectrogram ---
        # Using same settings as visualizer.py: n_mels=128, power=1.0 (amplitude)
        M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, power=1.0)
        M_db = librosa.amplitude_to_db(M, ref=np.max)
        
        ax_mel = axes[i, 1]
        img_mel = librosa.display.specshow(M_db, sr=sr, x_axis='time', y_axis='mel', ax=ax_mel)
        ax_mel.set_title(f"{instrument} - Mel Spectrogram")
        fig.colorbar(img_mel, ax=ax_mel, format='%+2.0f dB')

    output_path = os.path.join(OUTPUT_DIR, "instrument_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {output_path}")

if __name__ == "__main__":
    compare_instruments()
