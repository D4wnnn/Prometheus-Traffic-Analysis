"""
Main program for encrypted traffic dataset preprocessing
"""

import os
import sys
import argparse
from pathlib import Path
import logging
from datetime import datetime
import json
import shutil
from config import (LOG_DIR, SPLIT_MODE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, MIN_CATEGORY_FLOWS)
from utils import setup_logger, get_pcap_files
from pcap_processor import process_single_pcap

# Initialize main logger
log_file = LOG_DIR / f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger = setup_logger("main", log_file)


def clear_directory(dir_path: Path):
    """
    Clears all content within the directory if it exists and is not empty.
    """
    if dir_path.exists() and any(dir_path.iterdir()):
        logger.warning(f"Output directory '{dir_path}' is not empty, clearing contents...")
        for item in dir_path.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                    logger.info(f"Deleted subdirectory: {item}")
                else:
                    item.unlink()
                    logger.info(f"Deleted file: {item}")
            except Exception as e:
                logger.error(f"Failed to delete {item}: {e}")
        logger.info(f"Directory '{dir_path}' cleared successfully.")
    elif dir_path.exists():
        logger.info(f"Output directory '{dir_path}' is already empty.")
    else:
        logger.info(f"Output directory '{dir_path}' does not exist; it will be created later.")


class DatasetPreprocessor:
    """Main class for dataset preprocessing"""

    def __init__(self, input_dir, output_dir, train_ratio, val_ratio, test_ratio, min_category_flows=0, filter_rule="tcp or udp"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.filter_rule = filter_rule
        self.min_category_flows = min_category_flows  # Threshold for minimum flows per category
        self.stats = {
            "total_categories": 0,
            "total_pcaps": 0,
            "successful_pcaps": 0,
            "failed_pcaps": 0,
            "category_stats": {},
            "train_total": 0,
            "val_total": 0,
            "test_total": 0,
            "removed_categories": [],  # Track categories that fall below the threshold
        }

    def get_categories(self):
        """Retrieve all categories (subfolders)"""
        categories = [d for d in self.input_dir.iterdir() if d.is_dir()]
        return sorted(categories)

    def process_category(self, category_dir):
        """Process all PCAP files within a specific category"""
        category_name = category_dir.name
        logger.info("=" * 50)
        logger.info(f"Starting processing for category: {category_name}")

        # Get all PCAP files in the category
        pcap_files = get_pcap_files(category_dir)

        if not pcap_files:
            logger.warning(f"No PCAP files found for category {category_name}")
            return

        logger.info(f"Found {len(pcap_files)} PCAP files")

        success_count = 0
        fail_count = 0

        # Process each PCAP file
        for i, pcap_file in enumerate(pcap_files, 1):
            logger.info("-" * 40)
            logger.info(f"Progress: [{i}/{len(pcap_files)}] {pcap_file.name}")

            try:
                success = process_single_pcap(
                    category_name, 
                    pcap_file, 
                    self.output_dir, 
                    train_ratio=self.train_ratio,
                    val_ratio=self.val_ratio,
                    filter_rule=self.filter_rule
                )
                if success:
                    success_count += 1
                    self.stats["successful_pcaps"] += 1
                else:
                    fail_count += 1
                    self.stats["failed_pcaps"] += 1

            except Exception as e:
                logger.error(f"Error processing PCAP file: {e}")
                fail_count += 1
                self.stats["failed_pcaps"] += 1

        # Update category stats
        self.stats["category_stats"][category_name] = {
            "total": len(pcap_files),
            "success": success_count,
            "failed": fail_count,
        }

        logger.info(f"Category {category_name} completed: Success {success_count}/{len(pcap_files)}")

    def get_category_flow_count(self, category_name):
        """Calculate total flow count across all splits for a category"""
        train_cat_dir = self.output_dir / "train" / category_name
        val_cat_dir = self.output_dir / "val" / category_name
        test_cat_dir = self.output_dir / "test" / category_name

        total_flows = 0
        for cat_dir in [train_cat_dir, val_cat_dir, test_cat_dir]:
            if cat_dir.exists():
                flows = list(cat_dir.glob("*.pcap")) + list(cat_dir.glob("*.cap"))
                total_flows += len(flows)

        return total_flows

    def remove_small_categories(self):
        """Remove categories with flow counts below the threshold"""
        if self.min_category_flows <= 0:
            logger.info("MIN_CATEGORY_FLOWS <= 0, skipping category removal")
            return

        logger.info("=" * 50)
        logger.info(f"Checking for categories with fewer than {self.min_category_flows} flows...")

        train_dir = self.output_dir / "train"
        if not train_dir.exists():
            logger.warning("Training directory not found, skipping category removal")
            return

        categories_to_remove = []

        # Check each category
        for category_dir in train_dir.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                total_flows = self.get_category_flow_count(category_name)

                if total_flows < self.min_category_flows:
                    categories_to_remove.append((category_name, total_flows))
                    logger.warning(
                        f"Category '{category_name}' (flows: {total_flows}) is below threshold ({self.min_category_flows}) and will be removed."
                    )

        # Remove small categories
        for category_name, flow_count in categories_to_remove:
            try:
                for split in ["train", "val", "test"]:
                    cat_dir = self.output_dir / split / category_name
                    if cat_dir.exists():
                        shutil.rmtree(cat_dir)
                        logger.info(f"Deleted: {cat_dir}")

                # Log the removal
                self.stats["removed_categories"].append({
                    "name": category_name,
                    "flow_count": flow_count,
                    "reason": f"Flow count ({flow_count}) < threshold ({self.min_category_flows})"
                })

                if category_name in self.stats["category_stats"]:
                    del self.stats["category_stats"][category_name]

            except Exception as e:
                logger.error(f"Failed to remove category '{category_name}': {e}")

        if categories_to_remove:
            logger.info(f"Removed {len(categories_to_remove)} small categories in total.")
            self.stats["total_categories"] -= len(categories_to_remove)
        else:
            logger.info("No small categories found for removal.")

    def count_output_flows(self):
        """Count flow files in the final output directories"""
        splits = ["train", "val", "test"]
        counts = {"train": 0, "val": 0, "test": 0}

        for split in splits:
            split_dir = self.output_dir / split
            if split_dir.exists():
                for category_dir in split_dir.iterdir():
                    if category_dir.is_dir():
                        flows = list(category_dir.glob("*.pcap")) + list(category_dir.glob("*.cap"))
                        counts[split] += len(flows)

        self.stats["train_total"] = counts["train"]
        self.stats["val_total"] = counts["val"]
        self.stats["test_total"] = counts["test"]

        return counts["train"], counts["val"], counts["test"]

    def process_all(self):
        """Process all categories in the input directory"""
        categories = self.get_categories()

        if not categories:
            logger.error(f"No category folders found in input directory: {self.input_dir}")
            return

        logger.info(f"Found {len(categories)} categories")
        self.stats["total_categories"] = len(categories)

        for category_dir in categories:
            self.process_category(category_dir)
            self.stats["total_pcaps"] = (
                self.stats["successful_pcaps"] + self.stats["failed_pcaps"]
            )

        # Post-processing steps
        self.remove_small_categories()
        self.count_output_flows()
        self.print_stats()
        self.save_stats()

    def print_stats(self):
        """Print final processing statistics"""
        logger.info("=" * 50)
        logger.info("Processing Finished - Overall Statistics:")
        logger.info(f"  Total Categories: {self.stats['total_categories']}")
        logger.info(f"  Total PCAPs: {self.stats['total_pcaps']}")
        logger.info(f"  Processed Successfully: {self.stats['successful_pcaps']}")
        logger.info(f"  Failed: {self.stats['failed_pcaps']}")
        
        if self.stats["removed_categories"]:
            logger.info(f"\n  Removed Small Categories ({len(self.stats['removed_categories'])}):")
            for removed in self.stats["removed_categories"]:
                logger.info(f"    - {removed['name']}: {removed['reason']}")
        
        logger.info(f"\n  Output Flow Statistics:")
        logger.info(f"    Training Flows: {self.stats['train_total']}")
        logger.info(f"    Validation Flows: {self.stats['val_total']}")
        logger.info(f"    Test Flows: {self.stats['test_total']}")
        logger.info(
            f"    Total Flows: {self.stats['train_total'] + self.stats['val_total'] + self.stats['test_total']}"
        )

        logger.info("\nCategory Breakdown:")
        for category, cat_stats in self.stats["category_stats"].items():
            logger.info(f"  {category}: Success {cat_stats['success']}/{cat_stats['total']}")

    def save_stats(self):
        """Save statistics to a JSON file"""
        stats_file = self.output_dir / "preprocessing_stats.json"

        self.stats["output_structure"] = {
            "format": "train_val_test_split",
            "split_mode": SPLIT_MODE,
            "split_ratios": {
                "train": TRAIN_RATIO,
                "val": VAL_RATIO,
                "test": TEST_RATIO,
            },
            "min_category_flows": self.min_category_flows,
            "description": {
                "level1": "train/val/test split",
                "level2": "categories",
                "level3": "flow files with pcap prefix",
            },
        }

        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Statistics saved to: {stats_file}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Encrypted Traffic Dataset Preprocessing Tool")
    parser.add_argument("--input", type=str, help="Root directory of the input dataset")
    parser.add_argument("--output", type=str, help="Root directory for output")
    parser.add_argument("--category", type=str, help="Process only a specific category")
    parser.add_argument("--clear", action="store_true", help="Clear the output directory before starting")
    parser.add_argument("--min-category-flows", type=int, default=MIN_CATEGORY_FLOWS, help=f"Min flow threshold (default: {MIN_CATEGORY_FLOWS})")
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO, help=f"Train set ratio (default: {TRAIN_RATIO})")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO, help=f"Validation set ratio (default: {VAL_RATIO})")
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO, help=f"Test set ratio (default: {TEST_RATIO})")
    parser.add_argument("--filter-rule", type=str, default="tcp or udp", help="tshark filter rule (default: 'tcp or udp')")

    args = parser.parse_args()

    if not Path(args.input).exists():
        logger.error(f"Input directory does not exist: {args.input}")
        sys.exit(1)
    
    output_dir = Path(args.output)
    if args.clear:
        clear_directory(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info("Encrypted Traffic Dataset Preprocessing Tool")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Split Mode: {SPLIT_MODE}")
    logger.info(f"Ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    logger.info(f"Min Flows: {args.min_category_flows}")
    logger.info("=" * 50)

    preprocessor = DatasetPreprocessor(
        args.input, args.output, args.train_ratio, args.val_ratio, args.test_ratio, 
        min_category_flows=args.min_category_flows, filter_rule=args.filter_rule
    )

    if args.category:
        category_dir = Path(args.input) / args.category
        if category_dir.exists():
            preprocessor.process_category(category_dir)
            preprocessor.remove_small_categories()
            preprocessor.count_output_flows()
            preprocessor.print_stats()
            preprocessor.save_stats()
        else:
            logger.error(f"Specified category not found: {args.category}")
            sys.exit(1)
    else:
        preprocessor.process_all()

    logger.info("=" * 50)
    logger.info("All processing completed!")


if __name__ == "__main__":
    main()