import os
import subprocess
import csv
import sys
import argparse
import itertools

# Hyperparameter Grid
DROPOUT_RATES = [0.3, 0.4, 0.5]
LEARNING_RATES = [0.001, 0.0005, 0.0001]
L2_RATES = [0.001, 0.0001]

# Fixed parameters
EPOCHS = 15
BATCH_SIZE = 32
IMG_SIZE = 224

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_index", type=int, default=0, help="Index of this worker (0-based)")
    parser.add_argument("--total_shards", type=int, default=1, help="Total number of workers")
    args = parser.parse_args()

    # Generate full grid of combinations
    # Order: [(do, lr, l2), (do, lr, l2)...]
    full_grid = list(itertools.product(DROPOUT_RATES, LEARNING_RATES, L2_RATES))
    total_combinations = len(full_grid)
    
    # Calculate shard slice
    # Simple logic: distribute runs evenly
    # e.g., 18 runs, 3 shards -> 6 runs each
    chunk_size = (total_combinations + args.total_shards - 1) // args.total_shards
    start_idx = args.shard_index * chunk_size
    end_idx = min(start_idx + chunk_size, total_combinations)
    
    my_grid = full_grid[start_idx:end_idx]

    # Setup Output
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs", "tuning_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Each shard writes to its own CSV to avoid locking issues
    results_file = os.path.join(output_dir, f"tuning_shard_{args.shard_index}.csv")
    
    # Initialize CSV
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Run_ID', 'Dropout', 'LR', 'L2', 'Final_Val_Acc', 'Final_Val_Loss'])

    print(f"--- Worker {args.shard_index}/{args.total_shards} Started ---")
    print(f"Assigned {len(my_grid)} runs (Indices {start_idx} to {end_idx-1})")

    for i, (dropout, lr, l2) in enumerate(my_grid):
        global_run_id = start_idx + i + 1
        print(f"\n[Worker {args.shard_index}] Run {global_run_id}/{total_combinations}: DO={dropout}, LR={lr}, L2={l2}")
        
        run_output_dir = os.path.join(output_dir, f"run_{global_run_id}")
        os.makedirs(run_output_dir, exist_ok=True)
        
        cmd = [
            sys.executable, "src/training.py",
            "--epochs", str(EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--dropout", str(dropout),
            "--l2", str(l2),
            "--learning_rate", str(lr), 
            "--img_size", str(IMG_SIZE),
            "--output_dir", run_output_dir
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Simple parsing of the last validation metric
            # (In a real scenario, we might parse the JSON history if saved, but grep is fast)
            val_acc = "0.0"
            val_loss = "0.0"
            
            lines = result.stdout.split('\n')
            for line in reversed(lines):
                if "val_accuracy:" in line:
                    parts = line.replace(':', ' ').split()
                    try:
                        if "val_accuracy" in parts:
                            idx = parts.index("val_accuracy")
                            val_acc = parts[idx+1]
                        if "val_loss" in parts:
                            idx = parts.index("val_loss")
                            val_loss = parts[idx+1]
                        break
                    except: pass
            
            print(f"   -> Result: Acc={val_acc}, Loss={val_loss}")
            
            with open(results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([global_run_id, dropout, lr, l2, val_acc, val_loss])
                
        except subprocess.CalledProcessError as e:
            print(f"   -> Failed! Error: {e.stderr}")
            with open(results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([global_run_id, dropout, lr, l2, "FAILED", "FAILED"])

    print(f"--- Worker {args.shard_index} Completed ---")

if __name__ == "__main__":
    main()