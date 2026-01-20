from pathlib import Path
import tensorflow as tf

# Absolute model path
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "model"
    / "multilabel_cnn_improved.keras"
)

_MODEL = None  # singleton

def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = tf.keras.models.load_model(MODEL_PATH)
    return _MODEL
