import librosa
import numpy as np
from backend.model_loader import get_model

IMG_SIZE = 128
SR = 16000

def audio_to_mel(y, sr):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_db = np.resize(mel_db, (IMG_SIZE, IMG_SIZE))
    mel_db = mel_db / (np.max(np.abs(mel_db)) + 1e-8)

    # FORCE 3 CHANNELS (matches training)
    mel_db = np.stack([mel_db] * 3, axis=-1)

    return mel_db.astype("float32")


def predict_segments(mel_segments):
    model = get_model()
    return model.predict(mel_segments, verbose=0)
