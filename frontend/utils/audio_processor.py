import librosa
import numpy as np

class AudioProcessor:
    def __init__(self, sr=22050, n_mels=128):
        self.sr = sr
        self.n_mels = n_mels

    def load_audio(self, path):
        y, sr = librosa.load(path, sr=self.sr, mono=True)
        return y, sr

    def get_audio_info(self, path):
        y, sr = self.load_audio(path)
        return {
            "duration": len(y) / sr,
            "sample_rate": sr,
            "channels": 1
        }

    def compute_mel(self, y):
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_mels=self.n_mels
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db
