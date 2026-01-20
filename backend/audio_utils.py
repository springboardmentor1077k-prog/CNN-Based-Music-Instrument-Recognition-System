import librosa
import numpy as np

# -------------------------------
# AUDIO FEATURE EXTRACTION
# -------------------------------

def extract_features(
    audio_path,
    sr=22050,
    n_mels=128,
    max_len=128
):
    """
    Converts audio file → fixed-size Mel spectrogram
    Output shape: (1, 128, 128, 1)
    """

    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels
    )

    mel = librosa.power_to_db(mel, ref=np.max)

    # Pad / trim time axis
    if mel.shape[1] < max_len:
        pad_width = max_len - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad_width)))
    else:
        mel = mel[:, :max_len]

    # Normalize
    mel = (mel - mel.mean()) / (mel.std() + 1e-9)

    # Shape → (1, 128, 128, 1)
    mel = mel[np.newaxis, ..., np.newaxis]

    return mel
