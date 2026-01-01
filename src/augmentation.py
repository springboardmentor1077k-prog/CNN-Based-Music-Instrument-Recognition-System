import os

# Fix for Numba/Librosa permission issue in Docker (non-root)
os.environ['NUMBA_CACHE_DIR'] = '/tmp'

import librosa
import numpy as np

def add_noise(y, noise_level=0.005):
    """
    Adds Gaussian noise to the audio signal.
    """
    noise = np.random.randn(len(y))
    augmented_y = y + noise_level * noise
    return augmented_y

def time_stretch(y, rate=1.0):
    """
    Stretches the time of the audio signal without changing pitch.
    rate > 1.0 speeds up, rate < 1.0 slows down.
    """
    return librosa.effects.time_stretch(y, rate=rate)

def pitch_shift(y, sr, n_steps=0):
    """
    Shifts the pitch of the audio signal by n_steps semitones.
    """
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def adjust_volume(y, factor=1.0):
    """
    Adjusts the volume of the audio signal by a factor.
    factor > 1.0 increases volume, factor < 1.0 decreases it.
    """
    return y * factor

def time_shift(y, sr, shift_max=0.5):
    """
    Shifts the audio forward or backward by a random amount up to shift_max seconds.
    Wraps around (rolls).
    """
    shift = np.random.randint(int(sr * shift_max))
    return np.roll(y, shift)
