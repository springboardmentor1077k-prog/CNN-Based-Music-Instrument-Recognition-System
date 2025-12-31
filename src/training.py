import os
import sys
import argparse

# Fix for Numba/Librosa permission issue in Docker (non-root)
os.environ['NUMBA_CACHE_DIR'] = '/tmp'

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_trainer import ModelTrainer

def main():
    parser = argparse.ArgumentParser(description="Train InstruNet AI CNN Model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--l2", type=float, default=0.001, help="L2 regularization rate")
    parser.add_argument("--img_size", type=int, default=128, help="Input image size (height and width)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save outputs")
    
    args = parser.parse_args()

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN_DIR = os.path.join(PROJECT_ROOT, "datasets", "IRMAS-ProcessedTrainingData", "train", "spectrograms")
    VAL_DIR = os.path.join(PROJECT_ROOT, "datasets", "IRMAS-ProcessedTrainingData", "validation", "spectrograms")
    
    # Determine output directory
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    else:
        OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Ensure data exists
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print(f"Data directories not found.\nTrain: {TRAIN_DIR}\nVal: {VAL_DIR}")
        exit(1)

    print("--- Starting Model Training (HPC Optimized) ---")
    print(f"Configuration: Epochs={args.epochs}, Batch Size={args.batch_size}, Dropout={args.dropout}, L2={args.l2}")
    
    trainer = ModelTrainer(TRAIN_DIR, VAL_DIR, img_height=args.img_size, img_width=args.img_size, batch_size=args.batch_size)
    trainer.load_data()
    
    print(f"Building model with Dropout={args.dropout} and L2={args.l2}...")
    trainer.build_model(dropout_rate=args.dropout, l2_rate=args.l2)
    
    # Train
    trainer.train(epochs=args.epochs)
    
    # Plot History
    trainer.plot_history(OUTPUT_DIR)
    
    # Evaluate
    trainer.evaluate_model(OUTPUT_DIR)
    
    # Save Model
    trainer.save_model(OUTPUT_DIR)
    print("--- Training Completed ---")

if __name__ == "__main__":
    main()
