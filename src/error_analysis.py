import os
import sys

# Fix for Numba/Librosa permission issue
os.environ['NUMBA_CACHE_DIR'] = '/tmp'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import argparse

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="Error Analysis for InstruNet AI")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to validation spectrograms")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save analysis results")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--top_n", type=int, default=5, help="Number of worst predictions to analyze")
    args = parser.parse_args()

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Resolve Paths
    if args.data_dir:
        DATA_DIR = args.data_dir
    else:
        DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "IRMAS-ProcessedTrainingData", "validation", "spectrograms")
        
    if args.model_path:
        MODEL_PATH = args.model_path
    else:
        MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "instrunet_cnn.keras")
        
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    else:
        OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"--- Starting Error Analysis (Top {args.top_n} Worst Predictions) ---")
    
    # Load Model
    print("Loading model...")
    model = load_model(MODEL_PATH)

    # Load Validation Data
    print("Loading validation dataset...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        seed=123,
        image_size=(args.img_size, args.img_size),
        batch_size=32,
        shuffle=False 
    )
    class_names = val_ds.class_names
    file_paths = val_ds.file_paths 
    
    # Collect Predictions and True Labels
    y_true = []
    y_probs = []

    print("Generating predictions...")
    for images, labels in val_ds:
        if labels.ndim > 1: # One-hot encoded
            y_true.extend(np.argmax(labels.numpy(), axis=1))
        else:
            y_true.extend(labels.numpy())
        logits = model.predict(images, verbose=0)
        probs = tf.nn.softmax(logits).numpy()
        y_probs.extend(probs)

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    # Identify Mistakes
    mistakes = []
    for i in range(len(y_true)):
        true_idx = y_true[i]
        pred_idx = np.argmax(y_probs[i])
        confidence = y_probs[i][pred_idx]
        
        if true_idx != pred_idx:
            mistakes.append({
                'index': i,
                'file_path': file_paths[i],
                'true_label': class_names[true_idx],
                'pred_label': class_names[pred_idx],
                'confidence': confidence,
                'true_prob': y_probs[i][true_idx]
            })

    # Sort by Confidence (Descending) -> "Confident Mistakes"
    mistakes.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Top N Worst (for printing)
    top_mistakes = mistakes[:args.top_n]
    
    print(f"\nFound {len(mistakes)} total errors.")
    print(f"--- Top {args.top_n} Overconfident Mistakes ---")
    
    for m in top_mistakes:
        print(f"[{m['confidence']:.4f}] Predicted: {m['pred_label']} | Actual: {m['true_label']}")
        print(f"   File: {os.path.basename(m['file_path'])}")

    # Save ALL mistakes to CSV
    df = pd.DataFrame(mistakes)
    output_csv = os.path.join(OUTPUT_DIR, "error_analysis_all_mistakes.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nSaved FULL analysis of {len(mistakes)} errors to {output_csv}")
    print("---------------------------------------------------")

if __name__ == "__main__":
    main()
