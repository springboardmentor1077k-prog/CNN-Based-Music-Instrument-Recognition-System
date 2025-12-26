import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audio_processor import process_audio_file
from visualizer import save_clean_spectrogram

def load_true_labels(txt_path):
    """Reads the label file and returns a list of instrument codes."""
    if not os.path.exists(txt_path):
        return []
    
    with open(txt_path, 'r') as f:
        # Read lines, strip whitespace, and ignore empty lines
        labels = [line.strip() for line in f.readlines() if line.strip()]
    return labels

def main():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "IRMAS-TestingData")
    MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "instrunet_cnn.keras")
    TEMP_IMG_PATH = os.path.join(PROJECT_ROOT, "outputs", "temp_inference_spec.png")
    
    # Class names from training
    TRAIN_DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "IRMAS-TrainingData")
    # Ensure this matches the training order (alphabetical is standard for image_dataset_from_directory)
    if os.path.exists(TRAIN_DATA_DIR):
        CLASS_NAMES = sorted([d for d in os.listdir(TRAIN_DATA_DIR) if os.path.isdir(os.path.join(TRAIN_DATA_DIR, d))])
    else:
        # Fallback if training dir is missing, though it should exist
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
    
    for wav_file in tqdm(wav_files):
        file_path = os.path.join(TEST_DATA_DIR, wav_file)
        txt_path = os.path.join(TEST_DATA_DIR, wav_file.replace('.wav', '.txt'))
        
        # 1. Get True Labels & Filter
        true_labels = load_true_labels(txt_path)
        
        # SKIP if not exactly one label
        if len(true_labels) != 1:
            skipped_multi_label += 1
            continue
            
        true_label = true_labels[0]
        
        # Check if label is in our known classes (some test files might have unknown classes?)
        if true_label not in CLASS_NAMES:
            # print(f"Skipping unknown class: {true_label}")
            skipped_multi_label += 1 # Treating as skip
            continue

        # 2. Process Audio
        # Use same parameters as training: 16kHz, 3.0s
        y, sr = process_audio_file(file_path, target_sr=16000, duration=3.0)
        
        if y is None:
            errors += 1
            continue
            
        # 3. Generate Spectrogram Image (Identical to training pipeline)
        try:
            save_clean_spectrogram(y, sr, TEMP_IMG_PATH)
        except Exception as e:
            print(f"Error generating spectrogram for {wav_file}: {e}")
            errors += 1
            continue
            
        # 4. Load Image & Preprocess for Model
        try:
            # Target size must match training (128x128)
            img = image.load_img(TEMP_IMG_PATH, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create batch axis
            
            # CRITICAL: Rescaling (1./255). 
            # Note: In src/model_trainer.py, layers.Rescaling(1./255) is the FIRST layer of the model.
            # So we do NOT need to divide by 255 here manually if the model handles it.
            # Let's check model_trainer.py again... 
            # Yes: layers.Rescaling(1./255, input_shape=(self.img_height, self.img_width, 3)) is inside the model.
            # However, image_dataset_from_directory usually loads as float32 [0-255] or uint8 [0-255].
            # image.img_to_array returns float32 [0-255].
            # So passing [0-255] is correct because the model's first layer scales it.
            
        except Exception as e:
            print(f"Error processing image for {wav_file}: {e}")
            errors += 1
            continue
        
        # 5. Predict
        predictions = model.predict(img_array, verbose=0)
        predicted_index = np.argmax(predictions)
        predicted_label = CLASS_NAMES[predicted_index]
        
        # 6. Check Correctness
        if predicted_label == true_label:
            correct_predictions += 1
        
        total_evaluated += 1
        
    # Cleanup
    if os.path.exists(TEMP_IMG_PATH):
        os.remove(TEMP_IMG_PATH)
        
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
