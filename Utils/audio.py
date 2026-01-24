import numpy as np
import librosa

def load_audio(path, sr):
    y, _ = librosa.load(path, sr=sr, mono=False)
    return stereo_to_mono(y)

def stereo_to_mono(audio):
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=0)

def peak_normalize(audio):
    peak = np.max(np.abs(audio))
    return audio / peak if peak > 0 else audio

def trim_silence(audio, thresh=0.02):
    idx = np.where(np.abs(audio) > thresh)[0]
    if len(idx) == 0:
        return audio
    return audio[idx[0]:idx[-1]]

def fix_duration(audio, sr, duration):
    target_len = int(sr * duration)
    if len(audio) > target_len:
        return audio[:target_len]
    return np.pad(audio, (0, target_len - len(audio)))