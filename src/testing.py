import os
import sys
import argparse
import concurrent.futures

# Fix for Numba/Librosa permission issue in Docker (non-root)
os.environ['NUMBA_CACHE_DIR'] = '/tmp'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
import librosa

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audio_preprocessor import process_audio_file
from visualizer import save_clean_spectrogram

def load_true_labels(txt_path):
    """Reads the label file and returns a list of instrument codes."""
    if not os.path.exists(txt_path):
        return []
    
    with open(txt_path, 'r') as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    return labels

def prepare_file_windows(wav_file, test_data_dir, class_names):
    """Worker function to load audio and slice into windows."""
    file_path = os.path.join(test_data_dir, wav_file)
    txt_path = os.path.join(test_data_dir, wav_file.replace('.wav', '.txt'))
    
    true_labels = load_true_labels(txt_path)
    if len(true_labels) != 1 or true_labels[0] not in class_names:
        return None

    try:
        y_full, sr = librosa.load(file_path, sr=16000, mono=True)
    except Exception:
        return None

    window_size = int(3.0 * sr)
    stride = int(1.5 * sr)
    
    windows = []
    if len(y_full) < window_size:
        pad_width = window_size - len(y_full)
        windows.append(np.pad(y_full, (0, pad_width), mode='constant'))
    else:
        for start in range(0, len(y_full) - window_size + 1, stride):
            windows.append(y_full[start:start+window_size])
            
    return {
        'wav_file': wav_file,
        'true_label': true_labels[0],
        'windows': windows,
        'sr': sr
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate InstruNet AI on IRMAS Test Set")
    parser.add_argument("--test_dir", type=str, default=None, help="Path to IRMAS-TestingData")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained .keras model")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for temp images")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of audio files to batch process")
    args = parser.parse_args()

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TEST_DATA_DIR = args.test_dir or os.path.join(PROJECT_ROOT, "datasets", "IRMAS-TestingData")
    MODEL_PATH = args.model_path or os.path.join(PROJECT_ROOT, "outputs", "instrunet_cnn.keras")
    
    # Use output_dir for temp files if provided
    if args.output_dir:
        temp_dir = os.path.join(args.output_dir, "temp_inference")
    else:
        temp_dir = os.path.join(PROJECT_ROOT, "outputs", "temp_inference")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Class names from training
    TRAIN_DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "IRMAS-TrainingData")
    if os.path.exists(TRAIN_DATA_DIR):
        CLASS_NAMES = sorted([d for d in os.listdir(TRAIN_DATA_DIR) if os.path.isdir(os.path.join(TRAIN_DATA_DIR, d))])
    else:
        CLASS_NAMES = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
    
    print(f"--- IRMAS Single-Label Test Set Evaluation ---")
    print(f"Model: {MODEL_PATH}")
    print(f"Test Data: {TEST_DATA_DIR}")
    print(f"Classes: {CLASS_NAMES}")
    
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found. Please train the model first.")
        return

    # Load Model
    print("Loading model...")
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Find all WAV files
    wav_files = [f for f in os.listdir(TEST_DATA_DIR) if f.endswith('.wav')]
    
    correct_predictions = 0
    total_evaluated = 0
    skipped_multi_label = 0
    errors = 0
    
    print(f"Found {len(wav_files)} total audio files. Filtering for single-label samples...")
    
    # Process in large batches of files
    file_batch_size = args.batch_size
    
    for i in range(0, len(wav_files), file_batch_size):
        chunk = wav_files[i:i+file_batch_size]
        
        # 1. Parallel Load & Slice
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # We map the function to the chunk of files. 
            # Note: We pass fixed arguments (TEST_DATA_DIR, CLASS_NAMES) to the function
            batch_results = list(tqdm(executor.map(lambda f: prepare_file_windows(f, TEST_DATA_DIR, CLASS_NAMES), chunk), 
                                     total=len(chunk), desc=f"Batch {i//file_batch_size + 1}", leave=False))
        
        # Filter Nones (Skipped files or load errors)
        valid_results = [r for r in batch_results if r is not None]
        skipped_multi_label += (len(chunk) - len(valid_results))
        
        if not valid_results:
            continue

        # 2. Sequential Spectrogram Generation (Matplotlib is not thread-safe)
        all_windows = []
        file_window_counts = []
        
        for res in valid_results:
            count = 0
            for win in res['windows']:
                # Unique filename per window
                temp_path = os.path.join(temp_dir, f"{res['wav_file']}_win_{count}.png")
                save_clean_spectrogram(win, res['sr'], temp_path)
                
                try:
                    img = image.load_img(temp_path, target_size=(args.img_size, args.img_size))
                    all_windows.append(image.img_to_array(img))
                    os.remove(temp_path) # Cleanup immediately
                    count += 1
                except Exception as e:
                    print(f"Error processing window for {res['wav_file']}: {e}")
                    errors += 1
                    
            file_window_counts.append(count)

        # 3. Massive GPU Batch Prediction
        if not all_windows:
            continue
            
        all_windows_tensor = np.array(all_windows)
        all_predictions = model.predict(all_windows_tensor, batch_size=64, verbose=0)
        
        # 4. Soft Voting per File
        curr_idx = 0
        for j, res in enumerate(valid_results):
            count = file_window_counts[j]
            if count == 0:
                continue
                
            file_preds = all_predictions[curr_idx : curr_idx + count]
            curr_idx += count
            
            avg_pred = np.mean(file_preds, axis=0)
            predicted_index = np.argmax(avg_pred)
            predicted_label = CLASS_NAMES[predicted_index]
            
            if predicted_label == res['true_label']:
                correct_predictions += 1
            total_evaluated += 1
        
    # Cleanup temp dir
    try:
        os.rmdir(temp_dir)
    except:
        pass
        
    # Results
    print("\n" + "="*40)
    print("FINAL RESULTS (Single-Label Test Set)")
    print("="*40)
    print(f"Total Audio Files Found: {len(wav_files)}")
    print(f"Skipped (Multi-label / Unknown Class): {skipped_multi_label}")
    print(f"Total Evaluated (Single Label): {total_evaluated}")
    print(f"Processing Errors: {errors}")
    print("-" * 20)
    print(f"Correct Predictions: {correct_predictions}")
    
    if total_evaluated > 0:
        accuracy = (correct_predictions / total_evaluated) * 100
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("Accuracy: N/A (No samples evaluated)")
    print("="*40)

if __name__ == "__main__":
    main()
