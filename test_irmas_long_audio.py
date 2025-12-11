import tensorflow as tf
import numpy as np
import librosa
import os

# --- 1. CONFIGURATION (Do not delete this!) ---
MODEL_PATH = 'instrunet_poly_model.keras'

# ✅ UPDATED PATH: Pointing to your new Test_Audio folder
# Make sure the filename matches exactly what is in your folder
TEST_FILE_PATH = './Test_Audio/02 - Bo-Do-9.wav' 

# Variables causing your error
IMG_HEIGHT = 128
IMG_WIDTH = 128
SR = 16000
CHUNK_DURATION = 3.0 

# IRMAS Class mapping
READABLE_NAMES = ['Cello', 'Clarinet', 'Flute', 'Ac. Guitar', 'El. Guitar', 
                  'Organ', 'Piano', 'Saxophone', 'Trumpet', 'Violin', 'Voice']

def predict_long_audio(file_path):
    print(f"🎧 Analyzing: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        print("   Make sure you created the 'Test_Audio' folder and moved the file there.")
        return

    # 1. Load the FULL Audio
    try:
        y, sr = librosa.load(file_path, sr=SR)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    # Calculate chunks
    total_samples = len(y)
    chunk_samples = int(CHUNK_DURATION * SR) # 48000
    
    # Pad if too short
    if total_samples < chunk_samples:
        y = np.pad(y, (0, chunk_samples - total_samples))
        total_samples = len(y)

    print(f"   Duration: {total_samples/SR:.2f}s | Extracting 3s windows...")

    # 2. Sliding Window Prediction
    predictions = []
    hop_length = int(chunk_samples / 2) # 1.5s overlap
    
    # Load Model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except:
        print(f"❌ Error: Could not load {MODEL_PATH}. Did you run train_solo_fast.py?")
        return

    for start_idx in range(0, total_samples - chunk_samples + 1, hop_length):
        # Extract 3s Chunk
        end_idx = start_idx + chunk_samples
        chunk = y[start_idx:end_idx]
        
        # --- Preprocess Chunk ---
        # This is where your error was (IMG_HEIGHT is now defined above)
        spectrogram = librosa.feature.melspectrogram(y=chunk, sr=SR, n_mels=IMG_HEIGHT)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        
        # Normalize
        spectrogram_norm = (spectrogram_db + 80) / 80
        spectrogram_norm = np.clip(spectrogram_norm, 0, 1)
        
        # Ensure 128 width
        if spectrogram_norm.shape[1] < IMG_WIDTH:
            pad = IMG_WIDTH - spectrogram_norm.shape[1]
            spectrogram_norm = np.pad(spectrogram_norm, ((0,0), (0, pad)))
        else:
            spectrogram_norm = spectrogram_norm[:, :IMG_WIDTH]
            
        # Reshape
        input_data = np.expand_dims(spectrogram_norm, axis=-1)
        input_data = np.expand_dims(input_data, axis=0)
        
        # Predict
        pred = model.predict(input_data, verbose=0)
        predictions.append(pred[0])

    # 3. Average Predictions
    if not predictions:
        print("Error: Could not extract any segments.")
        return

    avg_prediction = np.mean(predictions, axis=0)

    # 4. Display Results
    print("\n--- FINAL SONG ANALYSIS ---")
    results = sorted(zip(READABLE_NAMES, avg_prediction), key=lambda x: x[1], reverse=True)
    
    for name, score in results:
        percentage = score * 100
        bar = "█" * int(score * 20)
        
        if percentage > 10: 
            print(f"👉 {name:<12} | {percentage:5.1f}% {bar}")
        else:
            print(f"   {name:<12} | {percentage:5.1f}%")

if __name__ == "__main__":
    predict_long_audio(TEST_FILE_PATH)