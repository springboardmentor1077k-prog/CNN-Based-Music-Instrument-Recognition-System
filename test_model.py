import tensorflow as tf
import numpy as np
import librosa
import os

# --- CONFIGURATION ---
MODEL_PATH = 'instrunet_solo_model.keras'
AUDIO_PATH = './Cleaned_Audio_Dataset' # We will pick a random file from here to test
IMG_HEIGHT = 128
IMG_WIDTH = 128
SR = 16000

# Defines the class names (Must match alphabetical order of folders)
# These are the standard IRMAS codes. You can rename them to English if you prefer.
CLASS_NAMES = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
# English Mapping: Cello, Clarinet, Flute, Acoustic Guitar, Electric Guitar, Organ, Piano, Saxophone, Trumpet, Violin, Voice

def preprocess_audio(file_path):
    """
    Loads audio and converts it to the exact format the model expects.
    """
    # 1. Load Audio
    y, sr = librosa.load(file_path, sr=SR)
    
    # 2. Generate Mel-Spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=IMG_HEIGHT)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    
    # 3. Normalize (-80dB...0dB -> 0...1) - MUST MATCH TRAINING LOGIC
    spectrogram_norm = (spectrogram_db + 80) / 80
    spectrogram_norm = np.clip(spectrogram_norm, 0, 1)
    
    # 4. Fix Width (Pad/Crop to 128)
    if spectrogram_norm.shape[1] < IMG_WIDTH:
        pad_width = IMG_WIDTH - spectrogram_norm.shape[1]
        spectrogram_norm = np.pad(spectrogram_norm, ((0,0), (0, pad_width)))
    else:
        spectrogram_norm = spectrogram_norm[:, :IMG_WIDTH]
        
    # 5. Add Batch Dimension (Model expects 1, 128, 128, 1)
    # The '1' at the end is the Channel (Grayscale)
    input_data = np.expand_dims(spectrogram_norm, axis=-1)
    input_data = np.expand_dims(input_data, axis=0)
    
    return input_data

def predict_instrument():
    print("--- 1. Loading Model ---")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model Loaded Successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print("--- 2. Selecting Random Test File ---")
    # Pick a random instrument folder
    random_class = np.random.choice(CLASS_NAMES)
    class_path = os.path.join(AUDIO_PATH, random_class)
    
    # Pick a random file inside
    files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
    if not files:
        print("No files found!")
        return
        
    random_file = np.random.choice(files)
    file_path = os.path.join(class_path, random_file)
    
    print(f"🎧 Testing File: {random_class} / {random_file}")

    # --- 3. PREDICTION ---
    input_data = preprocess_audio(file_path)
    
    # Get raw probabilities (e.g., [0.01, 0.9, 0.05...])
    predictions = model.predict(input_data)
    
    # Flatten array to simple list
    probs = predictions[0]

    print("\n--- 4. RESULTS ---")
    print(f"{'INSTRUMENT':<15} | {'CONFIDENCE':<10}")
    print("-" * 30)
    
    # Sort results by highest confidence
    # zip() combines names and scores, sorted() sorts them
    results = sorted(zip(CLASS_NAMES, probs), key=lambda x: x[1], reverse=True)
    
    for instrument, score in results:
        # Visualize bar
        bar_len = int(score * 20)
        bar = "█" * bar_len
        percentage = score * 100
        
        # Highlight if confidence > 20%
        if percentage > 20:
            print(f"👉 {instrument:<12} | {percentage:5.1f}% {bar}")
        else:
            print(f"   {instrument:<12} | {percentage:5.1f}% {bar}")

if __name__ == "__main__":
    predict_instrument()