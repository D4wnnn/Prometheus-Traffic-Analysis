import os
import shutil
import random
import argparse
from pathlib import Path

def split_dataset(input_dir, output_dir, split_ratios=(0.8, 0.1, 0.1), seed=42, clear_output=False, max_samples=None):
    """
    Randomly split the dataset according to ratios (Copy Mode).
    :param input_dir: Path to organized input directory.
    :param output_dir: Path to the target output directory.
    :param split_ratios: Tuple of (train, val, test) ratios.
    :param seed: Random seed for reproducibility.
    :param clear_output: Whether to clear the output directory before processing.
    :param max_samples: Maximum samples allowed per category (int); random downsampling applied if exceeded.
    """
    # 1. Set random seed for reproducibility
    random.seed(seed)
    
    # 2. Check and normalize ratios
    total_ratio = sum(split_ratios)
    train_r, val_r, test_r = [x / total_ratio for x in split_ratios]
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    # --- Logic: Clear output directory ---
    if clear_output:
        if output_path.exists():
            print(f"Warning: --clear flag detected. Deleting output directory: {output_dir}")
            try:
                shutil.rmtree(output_path)
                print("Output directory cleared.")
            except Exception as e:
                print(f"Error: Failed to clear directory - {e}")
                return
    # -------------------------------------

    # 3. Traverse first-level directories (Labels)
    # Get all subfolders, excluding hidden files
    categories = [d for d in input_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"{'='*20} Processing Started {'='*20}")
    print(f"Detected categories: {len(categories)}")
    print(f"Split ratios (Train/Val/Test): {train_r:.2f} / {val_r:.2f} / {test_r:.2f}")
    if max_samples:
        print(f"Max samples per category: {max_samples}")
    
    for category in categories:
        label_name = category.name
        
        # Get all files in this category (filtering for .pcap or .pcapng)
        files = [f for f in category.glob('*') if f.is_file() and f.suffix in ['.pcap', '.pcapng']]
        
        # If no pcap files are found, try to get all files
        if not files:
             files = [f for f in category.glob('*') if f.is_file()]
        
        file_count_original = len(files)
        if file_count_original == 0:
            print(f"Warning: Category {label_name} is empty, skipping.")
            continue
            
        # Shuffle (Crucial for both random splitting and downsampling)
        random.shuffle(files)
        
        # --- Logic: Max samples limit ---
        if max_samples is not None and file_count_original > max_samples:
            files = files[:max_samples] # Take the first max_samples
            print(f"Category [{label_name:<15}] -> Original: {file_count_original} | Exceeds limit, downsampling to: {max_samples}")
        
        file_count = len(files)
        # ---------------------------------
        
        # Calculate split indices
        train_end = int(file_count * train_r)
        val_end = train_end + int(file_count * val_r)
        
        # Split the list
        splits = {
            'train': files[:train_end],
            'val':   files[train_end:val_end],
            'test':  files[val_end:]
        }
        
        # Execute copying
        for split_name, split_files in splits.items():
            # Target directory structure: output/train/label_name/
            dest_dir = output_path / split_name / label_name
            os.makedirs(dest_dir, exist_ok=True)
            
            for file in split_files:
                try:
                    # copy2 preserves file metadata (timestamps, etc.)
                    shutil.copy2(file, dest_dir / file.name)
                except Exception as e:
                    print(f"Failed to copy {file.name}: {e}")
        
        # Print statistics for current category (reflects post-sampling distribution)
        print(f"Category [{label_name:<15}] -> Total Processed: {file_count:<5} "
              f"| Train: {len(splits['train']):<5} "
              f"| Val: {len(splits['val']):<5} "
              f"| Test: {len(splits['test']):<5}")

    print(f"{'='*20} Processing Complete {'='*20}")
    print(f"Dataset saved at: {output_dir}")

if __name__ == "__main__":
    # Use argparse for command line arguments
    parser = argparse.ArgumentParser(description="Randomly split PCAP dataset according to ratios.")
    
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to original dataset root directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output directory")
    parser.add_argument("--ratios", "-r", type=float, nargs=3, default=[0.75, 0.15, 0.15], 
                        help="Split ratios for Train, Val, Test (Default: 0.75 0.15 0.15)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed (Default: 42)")
    
    # --- Additional Parameters ---
    parser.add_argument("--clear", action="store_true", help="Clear the output directory before running")
    parser.add_argument("--max", type=int, default=None, help="Maximum number of samples to sample per category (Default: No limit)")
    # -----------------------------

    args = parser.parse_args()
    
    split_dataset(
        input_dir=args.input, 
        output_dir=args.output, 
        split_ratios=args.ratios, 
        seed=args.seed,
        clear_output=args.clear,
        max_samples=args.max
    )