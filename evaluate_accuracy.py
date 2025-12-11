import tensorflow as tf
import numpy as np
import librosa
import os
import glob

# --- CONFIGURATION ---
MODEL_PATH = 'instrunet_robust_model.keras'
# Point this to your Test_Audio folder (where you put the .wav and .txt files)
TEST_FOLDER = './Test_Audio' 
IMG_HEIGHT = 128
IMG_WIDTH = 128
SR = 16000
CHUNK_DURATION = 3.0 

# Must match the training order exactly (Alphabetical)
CLASS_NAMES = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']

def get_ground_truth(txt_path):
    """
    Reads the .txt file and returns a list of instruments present in the song.
    Example file content: "sax\ncla\n" -> Returns: ['sax', 'cla']
    """
    with open(txt_path, 'r') as f:
        # Read lines, strip whitespace, remove empty lines
        labels = [line.strip() for line in f.readlines() if line.strip()]
    return labels

def process_chunk_robust(y_chunk):
    """
    Same preprocessing logic as train_robust.py
    """
    spectrogram = librosa.feature.melspectrogram(y=y_chunk, sr=SR, n_mels=IMG_HEIGHT)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Normalize (-80db to 0db -> 0 to 1)
    spectrogram_norm = (spectrogram_db + 80) / 80
    spectrogram_norm = np.clip(spectrogram_norm, 0, 1)
    
    # Pad width
    if spectrogram_norm.shape[1] < IMG_WIDTH:
        pad_width = IMG_WIDTH - spectrogram_norm.shape[1]
        spectrogram_norm = np.pad(spectrogram_norm, ((0,0), (0, pad_width)))
    else:
        spectrogram_norm = spectrogram_norm[:, :IMG_WIDTH]
        
    return spectrogram_norm

def predict_song_average(model, wav_path):
    """
    Sliding window prediction for the whole song.
    Returns the average probability array.
    """
    try:
        y_full, _ = librosa.load(wav_path, sr=SR, mono=True)
        y_full = librosa.util.normalize(y_full)
    except Exception as e:
        print(f"❌ Error loading {wav_path}: {e}")
        return None

    total_samples = len(y_full)
    chunk_samples = int(CHUNK_DURATION * SR)
    hop_length = chunk_samples # Non-overlapping for speed
    
    predictions = []

    # Slide through the song
    for start_idx in range(0, total_samples, hop_length):
        end_idx = start_idx + chunk_samples
        chunk = y_full[start_idx:end_idx]
        
        # Pad last chunk if needed
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        
        # Preprocess
        input_data = process_chunk_robust(chunk)
        input_batch = np.expand_dims(input_data, axis=0)
        input_batch = np.expand_dims(input_batch, axis=-1)
        
        # Predict
        pred = model.predict(input_batch, verbose=0)[0]
        predictions.append(pred)

    if not predictions:
        return None
        
    # Average all chunks
    return np.mean(predictions, axis=0)

def evaluate_accuracy():
    print("--- 1. Loading Model ---")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file {MODEL_PATH} not found.")
        return
    
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print(f"--- 2. Scanning '{TEST_FOLDER}' ---")
    wav_files = glob.glob(os.path.join(TEST_FOLDER, "*.wav"))
    
    if not wav_files:
        print("❌ No .wav files found! Make sure you copied files from IRMAS-TestingData to 'Test_Audio'.")
        return

    correct_count = 0
    total_count = 0
    
    print(f"{'FILENAME':<30} | {'PREDICTED':<10} | {'TRUE LABELS':<15} | {'RESULT'}")
    print("-" * 75)

    for wav_path in wav_files:
        # Find matching .txt file
        txt_path = wav_path.replace('.wav', '.txt')
        
        if not os.path.exists(txt_path):
            print(f"⚠️ Warning: No label file for {os.path.basename(wav_path)}. Skipping.")
            continue
            
        # 1. Get Ground Truth
        true_labels = get_ground_truth(txt_path)
        
        # 2. Get Model Prediction
        avg_probs = predict_song_average(model, wav_path)
        if avg_probs is None: continue
        
        # Find the winner (highest score)
        top_index = np.argmax(avg_probs)
        predicted_label = CLASS_NAMES[top_index]
        confidence = avg_probs[top_index] * 100
        
        # 3. Check if correct
        # It counts as correct if the predicted instrument is ANY of the true instruments
        is_correct = predicted_label in true_labels
        
        if is_correct:
            correct_count += 1
            result_icon = "✅"
        else:
            result_icon = "❌"
            
        total_count += 1
        
        # Print row
        fname = os.path.basename(wav_path)[:28] # Truncate for display
        true_str = ",".join(true_labels)
        print(f"{fname:<30} | {predicted_label} ({int(confidence)}%) | {true_str:<15} | {result_icon}")

    # --- FINAL REPORT ---
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        print("-" * 75)
        print(f"\n📊 FINAL ACCURACY: {accuracy:.2f}%")
        print(f"   Correctly Identified: {correct_count} / {total_count} songs")
        print("\nNote: Accuracy is based on whether the Top-1 prediction appears in the song's label list.")
    else:
        print("\n❌ No valid test pairs (wav + txt) found.")

if __name__ == "__main__":
    evaluate_accuracy()