import os
import shutil
from tqdm import tqdm

def organize_dataset():
    # ================= Configuration Paths =================
    # Base path (set according to your environment)
    base_path = "path/to/your/dataset"  # Please modify to the actual path
    
    # Directory where original pcap files are located
    source_dir = os.path.join(base_path, "real_data")
    
    # Label file path
    label_file_path = os.path.join(base_path, "part1_label.txt")
    
    # Output directory (where organized data will be stored)
    output_dir = os.path.join(base_path, "organized_real_data")
    # =======================================================

    # 1. Read label file
    print(f"Reading label file: {label_file_path} ...")
    file_label_map = {}
    try:
        with open(label_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    filename = parts[0]
                    label = parts[1]
                    file_label_map[filename] = label
    except FileNotFoundError:
        print(f"Error: Label file not found at {label_file_path}")
        return

    print(f"Found {len(file_label_map)} sample mapping relations.")

    # 2. Start organizing
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    success_count = 0
    missing_count = 0

    # Use tqdm to display progress bar
    print("Starting to copy and organize files...")
    for filename, label in tqdm(file_label_map.items()):
        src_path = os.path.join(source_dir, filename)
        
        # Target folder path (first-level directory is the label)
        dst_folder = os.path.join(output_dir, label)
        dst_path = os.path.join(dst_folder, filename)

        # Check if source file exists
        if os.path.exists(src_path):
            # Create target label folder if it doesn't exist
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)
            
            # Copy file (using copy2 to preserve metadata)
            shutil.copy2(src_path, dst_path)
            success_count += 1
        else:
            # print(f"Warning: Source file {filename} not found in {source_dir}")
            missing_count += 1

    print("-" * 30)
    print("Organization complete!")
    print(f"Successfully organized: {success_count} files")
    if missing_count > 0:
        print(f"Missing files: {missing_count} (In label list but not found in directory)")
    print(f"New dataset saved at: {output_dir}")

if __name__ == "__main__":
    organize_dataset()