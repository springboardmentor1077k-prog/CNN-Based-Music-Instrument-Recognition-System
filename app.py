import streamlit as st
import tensorflow as tf
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# Ensure this points to the model created by train_robust.py
MODEL_PATH = 'vk_multilabel_model.keras' 
IMG_HEIGHT = 128
IMG_WIDTH = 128
SR = 16000
CHUNK_DURATION = 3.0 

READABLE_NAMES = ['Cello', 'Clarinet', 'Flute', 'Ac. Guitar', 'El. Guitar', 
                  'Organ', 'Piano', 'Saxophone', 'Trumpet', 'Violin', 'Voice']

st.set_page_config(page_title="InstruNet AI", layout="wide")
st.title("🎵 InstruNet AI: Music Instrument Recognition")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Confidence Threshold (%)", 0, 100, 20)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def process_chunk(y_chunk):
    """
    Exact match for train_robust.py preprocessing.
    """
    # 1. Mel-Spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y_chunk, sr=SR, n_mels=IMG_HEIGHT)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    
    # 2. Normalize Math (-80dB to 0dB -> 0.0 to 1.0)
    spectrogram_norm = (spectrogram_db + 80) / 80
    spectrogram_norm = np.clip(spectrogram_norm, 0, 1)
    
    # 3. Pad Width to 128
    if spectrogram_norm.shape[1] < IMG_WIDTH:
        pad_width = IMG_WIDTH - spectrogram_norm.shape[1]
        spectrogram_norm = np.pad(spectrogram_norm, ((0,0), (0, pad_width)))
    else:
        spectrogram_norm = spectrogram_norm[:, :IMG_WIDTH]
        
    return spectrogram_norm, spectrogram_db

def process_long_audio(uploaded_file):
    # Load and Normalize Audio
    y_full, sr = librosa.load(uploaded_file, sr=SR, mono=True)
    y_full = librosa.util.normalize(y_full)
    
    total_samples = len(y_full)
    chunk_samples = int(CHUNK_DURATION * SR)
    hop_length = chunk_samples 
    
    all_predictions = []
    viz_spectrogram = None
    
    # Sliding Window
    progress_bar = st.progress(0)
    chunks_count = int(np.ceil(total_samples / hop_length))
    
    for i, start_idx in enumerate(range(0, total_samples, hop_length)):
        end_idx = start_idx + chunk_samples
        chunk = y_full[start_idx:end_idx]
        
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            
        # Use the Math Function
        spec_norm, spec_db = process_chunk(chunk)
        
        # Save one sample for visualization
        if viz_spectrogram is None and np.max(spec_db) > -70: 
            viz_spectrogram = spec_db

        # Predict
        input_batch = np.expand_dims(spec_norm, axis=0) # Add batch
        input_batch = np.expand_dims(input_batch, axis=-1) # Add channel
        
        pred = model.predict(input_batch, verbose=0)[0]
        all_predictions.append(pred)
        
        if chunks_count > 0:
            progress_bar.progress(min((i+1)/chunks_count, 1.0))

    avg_prediction = np.mean(all_predictions, axis=0)
    if viz_spectrogram is None: viz_spectrogram = np.zeros((128, 128))

    return y_full, avg_prediction, viz_spectrogram

uploaded_file = st.file_uploader("Upload Audio", type=['wav', 'mp3'])

if uploaded_file is not None:
    model = load_model()
    
    with st.spinner("Analyzing Spectrogram Data..."):
        y_full, preds, spec_viz = process_long_audio(uploaded_file)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("1. Mel-Spectrogram Analysis")
        st.write("Visual representation of the input features:")
        
        # DISPLAY THE SPECTROGRAM HERE
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(spec_viz, sr=SR, x_axis='time', y_axis='mel', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        st.pyplot(fig)
        
        st.audio(uploaded_file, format='audio/wav')

    with col2:
        st.subheader("2. Detected Instruments")
        results = sorted(zip(READABLE_NAMES, preds), key=lambda x: x[1], reverse=True)
        
        found = False
        for name, score in results:
            percentage = score * 100
            if percentage >= threshold:
                found = True
                st.write(f"**{name}**")
                st.progress(int(percentage))
                st.caption(f"{percentage:.1f}% Confidence")
        
        if not found:
            st.warning("No instruments detected above threshold.")