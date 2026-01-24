import streamlit as st
import gdown
from pathlib import Path
import tensorflow as tf

MODEL_PATH = Path("model/multilabel_cnn_improved.keras")
MODEL_URL = "https://drive.google.com/uc?export=download&id=1WaTwAIm3cENjgL1o_khxyVqWy-jZBPZ8"

_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        if not MODEL_PATH.exists():
            with st.spinner("Downloading model (first run only)..."):
                MODEL_PATH.parent.mkdir(exist_ok=True)
                gdown.download(MODEL_URL, str(MODEL_PATH), quiet=False)

        with st.spinner("Loading AI model..."):
            _MODEL = tf.keras.models.load_model(MODEL_PATH)

    return _MODEL
