import librosa
import numpy as np

SR = 16000
N_MELS = 128
IMG_SIZE = 128
HOP_SEC = 1.0
WIN_SEC = 2.0

def audio_to_mel(y, sr):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = librosa.util.fix_length(mel, IMG_SIZE, axis=1)
    mel = mel[:IMG_SIZE, :]
    mel = mel / (abs(mel).max() + 1e-6)
    return mel.astype("float32")

def make_3channel(mel):
    delta = librosa.feature.delta(mel)
    delta2 = librosa.feature.delta(mel, order=2)
    return np.stack([mel, delta, delta2], axis=-1)

def segment_audio(y, sr):
    hop = int(HOP_SEC * sr)
    win = int(WIN_SEC * sr)

    segments = []
    for start in range(0, len(y) - win, hop):
        seg = y[start:start + win]
        mel = audio_to_mel(seg, sr)
        mel = make_3channel(mel)
        segments.append(mel)

    return np.array(segments)
