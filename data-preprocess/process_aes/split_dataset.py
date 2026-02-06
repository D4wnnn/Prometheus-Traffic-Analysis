import os
import shutil
import random
import argparse
from pathlib import Path

def split_dataset(input_dir, output_dir, train_ratio, val_ratio, test_ratio, seed=42):
    # 1. Validate the sum of ratios
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5:
        print("Error: The sum of train, val, and test ratios must equal 1.0")
        return

    # Set random seed for reproducibility
    random.seed(seed)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Check if the input directory exists
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    # Get all first-level directories as Labels
    labels = [d for d in input_path.iterdir() if d.is_dir()]
    
    if not labels:
        print("No label directories found. Please check the input directory structure.")
        return

    print(f"Found {len(labels)} classes (Labels). Starting processing...")

    # Define split names
    splits = ['train', 'val', 'test']

    # Iterate through each Label for splitting
    for label_dir in labels:
        label_name = label_dir.name
        # Get all pcap files under the current Label
        # If the extension is different, modify the glob pattern (e.g., '*.pcapng')
        files = list(label_dir.glob('*.pcap'))
        
        # Shuffle the file list randomly
        random.shuffle(files)
        
        total_files = len(files)
        if total_files == 0:
            print(f"Skipping empty directory: {label_name}")
            continue

        # Calculate split indices
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)

        # Split data into sets
        split_files = {
            'train': files[:train_end],
            'val': files[train_end:val_end],
            'test': files[val_end:]
        }

        print(f"Processing Label: {label_name} (Total: {total_files}) -> "
              f"Train: {len(split_files['train'])}, "
              f"Val: {len(split_files['val'])}, "
              f"Test: {len(split_files['test'])}")

        # Execute file copying
        for split_name, file_list in split_files.items():
            # Target directory structure: output/split_name/label_name/
            target_dir = output_path / split_name / label_name
            target_dir.mkdir(parents=True, exist_ok=True)

            for file_path in file_list:
                # Copy file
                # shutil.copy2 preserves metadata (e.g., timestamps). 
                # Use shutil.copy if metadata preservation is not required.
                shutil.copy2(file_path, target_dir / file_path.name)

    print("\n--- Dataset splitting completed ---")
    print(f"Output directory: {output_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly split traffic classification dataset (Pcap)")
    
    # Arguments
    parser.add_argument('--input', type=str, required=True, help='Path to the root directory of the original dataset')
    parser.add_argument('--output', type=str, required=True, help='Path to the root directory of the output dataset')
    parser.add_argument('--train', type=float, default=0.7, help='Ratio for training set (default: 0.7)')
    parser.add_argument('--val', type=float, default=0.15, help='Ratio for validation set (default: 0.15)')
    parser.add_argument('--test', type=float, default=0.15, help='Ratio for test set (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    split_dataset(
        input_dir=args.input,
        output_dir=args.output,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed
    )