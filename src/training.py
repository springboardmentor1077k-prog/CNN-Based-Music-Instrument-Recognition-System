import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_trainer import ModelTrainer

def main():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "IRMAS-ProcessedTrainingData", "spectrograms")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Ensure data exists
    if not os.path.exists(DATA_DIR):
        print(f"Data directory not found: {DATA_DIR}")
        exit(1)

    print("--- Starting Task 8: Model Training ---")
    trainer = ModelTrainer(DATA_DIR, batch_size=32)
    trainer.load_data()
    trainer.build_model()
    
    # Train for 20 epochs
    trainer.train(epochs=20)
    
    # Plot History
    trainer.plot_history(OUTPUT_DIR)
    
    # Evaluate
    trainer.evaluate_model(OUTPUT_DIR)
    
    # Save Model
    trainer.save_model(OUTPUT_DIR)
    print("--- Task 8 Completed ---")

if __name__ == "__main__":
    main()
