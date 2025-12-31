import os
import sys

# Fix for Numba/Librosa permission issue in Docker (non-root)
os.environ['NUMBA_CACHE_DIR'] = '/tmp'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use('Agg')  # Required for headless plotting on HPC
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    precision_recall_fscore_support, 
    roc_curve, 
    auc, 
    confusion_matrix, 
    classification_report
)
from sklearn.preprocessing import label_binarize

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def plot_roc_curve(y_true_bin, y_score, class_names, output_dir):
    """Plots One-vs-Rest ROC curves for each class."""
    n_classes = len(class_names)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(12, 8))
    colors = plt.cm.get_cmap('tab20', n_classes)
    
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=colors(i), lw=2,
                 label=f'ROC curve of {class_names[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Multi-class')
    plt.legend(loc="lower right")
    
    roc_path = os.path.join(output_dir, 'roc_curves.png')
    plt.savefig(roc_path)
    plt.close()
    print(f"ROC curves saved to {roc_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Detailed Evaluation of InstruNet AI")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to validation spectrograms")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation results")
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

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Model
    print("Loading model...")
    model = load_model(MODEL_PATH)

    # 2. Load Validation Data
    print("Loading validation dataset...")
    # We use same parameters as training to ensure consistency
    # Data is already pre-split into 'validation' folder by preprocessing.py
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        seed=123,
        image_size=(128, 128),
        batch_size=32,
        shuffle=False # CRITICAL: Do not shuffle for metrics
    )
    class_names = val_ds.class_names
    
    # 3. Generate Predictions
    print("Generating predictions on validation data...")
    y_true = []
    y_probs = []
    
    # Iterate over dataset to get labels and scores
    for images, labels in val_ds:
        y_true.extend(labels.numpy())
        logits = model.predict(images, verbose=0)
        # Convert logits to probabilities using softmax
        probs = tf.nn.softmax(logits).numpy()
        y_probs.extend(probs)
        
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = np.argmax(y_probs, axis=1)

    # 4. Detailed Metrics (Precision, Recall, F1)
    print("Calculating detailed metrics...")
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(len(class_names)))
    
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    metrics_path = os.path.join(OUTPUT_DIR, 'detailed_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Detailed metrics saved to {metrics_path}")

    # 5. Probability Threshold Analysis (Example for first class)
    # Thresholding probability into binary prediction (Task 10 requirement)
    threshold = 0.5
    y_pred_thresholded = (y_probs > threshold).astype(int)
    # Save a sample of probability to binary conversion
    sample_probs = pd.DataFrame(y_probs[:5], columns=class_names)
    sample_binary = pd.DataFrame(y_pred_thresholded[:5], columns=class_names)
    print("\nSample Probabilities (First 5):")
    print(sample_probs)
    print(f"\nSample Binary Predictions (Threshold > {threshold}):")
    print(sample_binary)

    # 6. Per-Class Confusion Matrix (Already done in Task 9, but can refine if needed)
    # Task 10 specifically asks for "Compute per class confusion matrix"
    # Usually this means normalized or visual
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(OUTPUT_DIR, 'normalized_confusion_matrix.png'))
    plt.close()

    # 7. Plot ROC Curves
    print("Plotting ROC curves...")
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    plot_roc_curve(y_true_bin, y_probs, class_names, OUTPUT_DIR)

    print("\n--- Task 10 Evaluation Completed ---")

if __name__ == "__main__":
    main()
