import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_mel_spectrogram_power(y, sr, file_name, output_dir):
    """Generates Mel Spectrogram using power_to_db (Correct for default librosa output)."""
    plt.figure(figsize=(12, 4))
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    M_db = librosa.power_to_db(M, ref=np.max)
    librosa.display.specshow(M_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram (Power) of {file_name}')
    plt.tight_layout()
    mel_path = os.path.join(output_dir, f"mel_spectrogram_power_{file_name}.png")
    plt.savefig(mel_path, dpi=300)
    plt.close()
    print(f"  - Mel Spectrogram (Power) saved to {mel_path}")

def plot_mel_spectrogram_amplitude(y, sr, file_name, output_dir):
    """Generates Mel Spectrogram using amplitude_to_db (Incorrect if input is Power, but kept for redundancy)."""
    plt.figure(figsize=(12, 4))
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    M_db = librosa.amplitude_to_db(M, ref=np.max)
    librosa.display.specshow(M_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram (Amplitude) of {file_name}')
    plt.tight_layout()
    mel_path = os.path.join(output_dir, f"mel_spectrogram_amplitude_{file_name}.png")
    plt.savefig(mel_path, dpi=300)
    plt.close()
    print(f"  - Mel Spectrogram (Amplitude) saved to {mel_path}")

def plot_spectrograms(y, sr, file_name, output_dir):
    """
    Generates and plots STFT spectrogram, Mel spectrogram, and MFCCs.

    Args:
        y (np.ndarray): The audio time series.
        sr (int): Sampling rate of `y`.
        file_name (str): Original file name for titling plots.
        output_dir (str): Directory to save the plots.
    """
    print(f"Generating spectrograms for {file_name}...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. STFT Spectrogram
    plt.figure(figsize=(12, 4))
    # STFT
    D = librosa.stft(y)
    # Amplitude to dB
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'STFT Spectrogram of {file_name}')
    plt.tight_layout()
    stft_path = os.path.join(output_dir, f"stft_spectrogram_{file_name}.png")
    plt.savefig(stft_path, dpi=300)
    plt.close()
    print(f"  - STFT Spectrogram saved to {stft_path}")

    # 2. Mel Spectrogram (Using Power by default)
    plot_mel_spectrogram_power(y, sr, file_name, output_dir)
    # Uncomment below to generate the amplitude version as well
    # plot_mel_spectrogram_amplitude(y, sr, file_name, output_dir)

    # 3. MFCCs
    plt.figure(figsize=(12, 4))
    # n_mfcc=13 is standard for speech/audio classification
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title(f'MFCCs of {file_name}')
    plt.tight_layout()
    mfcc_path = os.path.join(output_dir, f"mfcc_{file_name}.png")
    plt.savefig(mfcc_path, dpi=300)
    plt.close()
    print(f"  - MFCCs saved to {mfcc_path}")
