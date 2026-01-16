import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from audio_preprocessor import process_audio_file
from visualizer import save_mel_spectrogram

def main():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Target file
    # Source Audio
    AUDIO_PATH = os.path.join(PROJECT_ROOT, "datasets", "IRMAS-TrainingData", "cel", "008__[cel][nod][cla]0058__1.wav")
    # Existing Training Image (Ground Truth for what the model expects)
    TRAIN_IMG_PATH = os.path.join(PROJECT_ROOT, "datasets", "IRMAS-ProcessedTrainingData", "spectrograms", "cel", "008__[cel][nod][cla]0058__1.png")
    # New Inference Image (What we are generating now)
    DEBUG_IMG_PATH = os.path.join(PROJECT_ROOT, "outputs", "debug_inference_spec.png")
    
    print(f"--- Debugging Spectrogram Generation ---")
    
    # 1. Generate new spectrogram using current pipeline
    print("Processing audio...")
    y, sr = process_audio_file(AUDIO_PATH, target_sr=16000, duration=3.0)
    
    print("Generating inference spectrogram...")
    # NOTE: In evaluate_test_set.py, we called: 
    # save_mel_spectrogram(y, sr, TEMP_IMG_PATH, title=f"Mel Spectrogram (Amp) - {wav_file}")
    # The title changes based on filename! This adds different text to the image!
    
    # Let's try to mimic EXACTLY what evaluate_test_set does
    fake_filename = "008__[cel][nod][cla]0058__1.wav"
    save_mel_spectrogram(y, sr, DEBUG_IMG_PATH, title=f"Mel Spectrogram (Amp) - {fake_filename}")
    
    # 2. Compare Side-by-Side
    print("Creating comparison plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Load and display Training Image
    if os.path.exists(TRAIN_IMG_PATH):
        img_train = mpimg.imread(TRAIN_IMG_PATH)
        axes[0].imshow(img_train)
        axes[0].set_title(f"Training Image\n(Shape: {img_train.shape})")
        axes[0].axis('off')
    else:
        print(f"Warning: Training image not found at {TRAIN_IMG_PATH}")
    
    # Load and display Inference Image
    img_debug = mpimg.imread(DEBUG_IMG_PATH)
    axes[1].imshow(img_debug)
    axes[1].set_title(f"Inference Image\n(Shape: {img_debug.shape})")
    axes[1].axis('off')
    
    comparison_path = os.path.join(PROJECT_ROOT, "outputs", "debug_comparison.png")
    plt.savefig(comparison_path)
    print(f"Comparison saved to {comparison_path}")
    
    # 3. Print Image Details
    print(f"\nImage Analysis:")
    if os.path.exists(TRAIN_IMG_PATH):
        print(f"Training Image Shape: {img_train.shape} (Height, Width, Channels)")
    print(f"Inference Image Shape: {img_debug.shape}")

if __name__ == "__main__":
    main()
