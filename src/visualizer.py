import os

# Fix for Numba/Librosa permission issue in Docker (non-root)
os.environ['NUMBA_CACHE_DIR'] = '/tmp'

import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Required for headless plotting on HPC
import matplotlib.pyplot as plt
import numpy as np
import os

def save_mel_spectrogram(y, sr, output_path, title=None):
    """
    Generates and saves a Mel Spectrogram (Amplitude) plot to the specified path.
    """
    if title is None:
        title = "Mel Spectrogram (Amplitude)"
        
    plt.figure(figsize=(12, 4))
    # power=1.0 for magnitude (amplitude) spectrogram
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, power=1.0)
    M_db = librosa.amplitude_to_db(M, ref=np.max)
    librosa.display.specshow(M_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def save_clean_spectrogram(y, sr, output_path):
    """
    Generates and saves a CLEAN Mel Spectrogram (no axes, no labels, no title).
    Crucial for training CNNs to prevent overfitting on visual artifacts.
    """
    plt.figure(figsize=(12, 4))
    # Remove all margins and axes
    plt.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, power=1.0)
    M_db = librosa.amplitude_to_db(M, ref=np.max)
    
    librosa.display.specshow(M_db, sr=sr)
    
    # Save without any padding or borders
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_hpss_image(y, sr, output_path):
    """
    Performs HPSS (Harmonic-Percussive Source Separation) and saves a 
    3-Channel RGB image:
    - Red Channel: Harmonic Component
    - Green Channel: Percussive Component
    - Blue Channel: Original Spectrogram
    
    This creates a "data-rich" image for the CNN, allowing it to see 
    timbre and rhythm separately.
    """
    # 1. Decompose Audio
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # 2. Compute Mel Spectrograms (in dB)
    def get_db_spec(audio_data):
        m = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
        # Convert to dB, max is 0 dB
        return librosa.amplitude_to_db(m, ref=np.max)

    db_h = get_db_spec(y_harmonic)
    db_p = get_db_spec(y_percussive)
    db_o = get_db_spec(y)

    # 3. Normalize to [0, 1] range for Image format
    # We assume a dynamic range of 80 dB (standard for audio)
    min_db = -80.0
    max_db = 0.0
    
    def normalize(S):
        return np.clip((S - min_db) / (max_db - min_db), 0, 1)

    norm_h = normalize(db_h)
    norm_p = normalize(db_p)
    norm_o = normalize(db_o)

    # 4. Stack into RGB Image (Height, Width, 3)
    # librosa outputs (n_mels, time). Image expects (Height, Width, Channels).
    # We flip vertically so low frequencies are at the bottom.
    img_stack = np.dstack((np.flipud(norm_h), np.flipud(norm_p), np.flipud(norm_o)))
    
    # 5. Save directly using matplotlib's image save (handles the array->png conversion)
    plt.imsave(output_path, img_stack)

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

    # 2. Mel Spectrogram (Using Amplitude)
    mel_path = os.path.join(output_dir, f"mel_spectrogram_amplitude_{file_name}.png")
    save_mel_spectrogram(y, sr, mel_path, title=f'Mel Spectrogram (Amplitude) of {file_name}')
    print(f"  - Mel Spectrogram (Amplitude) saved to {mel_path}")

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
