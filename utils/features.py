# utils/features.py

import librosa
import numpy as np
from config import N_MELS, TARGET_FRAMES, EPS

def generate_log_mel(audio, sr):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        n_mels=N_MELS,
        power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return (mel_db - mel_db.mean()) / (mel_db.std() + EPS)

def fix_mel_frames(mel):
    if mel.shape[1] < TARGET_FRAMES:
        mel = np.pad(mel, ((0,0),(0, TARGET_FRAMES - mel.shape[1])))
    return mel[:, :TARGET_FRAMES]
