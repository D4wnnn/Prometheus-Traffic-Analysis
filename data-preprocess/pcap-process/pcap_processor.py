import os
import shutil
from pathlib import Path
import tempfile
import logging
from datetime import datetime

from utils import run_tshark, run_splitcap, clean_directory
from flow_utils import (
    collect_flows,
    split_flows_by_time,
    split_flows_randomly,
    save_flows_to_directory_with_prefix,
)
from config import SPLIT_MODE, RANDOM_SEED, TEMP_DIR

logger = logging.getLogger(__name__)


class PcapProcessor:
    """Class to process a single PCAP file"""

    def __init__(self, category_name, pcap_path, output_base_dir, train_ratio=0.7, val_ratio=0.15, filter_rule="tcp or udp"):
        self.category_name = category_name
        self.pcap_path = Path(pcap_path)
        self.pcap_name = self.pcap_path.stem  # Filename without extension
        self.output_base_dir = Path(output_base_dir)
        self.train_ratio = train_ratio 
        self.val_ratio = val_ratio
        
        # Output directory structure: train/val/test -> category -> pcap_files
        self.train_dir = self.output_base_dir / "train" / category_name
        self.val_dir = self.output_base_dir / "val" / category_name
        self.test_dir = self.output_base_dir / "test" / category_name

        # Temporary directories
        self.temp_base = TEMP_DIR / f"{category_name}_{self.pcap_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.filtered_pcap = self.temp_base / "filtered.pcap"
        self.flows_dir = self.temp_base / "flows"
        self.filter_rule = filter_rule

    def setup_directories(self):
        """Create necessary directories"""
        for dir_path in [self.train_dir, self.val_dir, self.test_dir, self.temp_base, self.flows_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def cleanup_temp(self):
        """Clean up temporary files"""
        if self.temp_base.exists():
            shutil.rmtree(self.temp_base)

    def process(self):
        """Main workflow for processing the PCAP file"""
        logger.info(f"Starting process: {self.category_name}/{self.pcap_name}")

        try:
            # 1. Create directories
            self.setup_directories()

            # 2. Filter traffic
            logger.info(f"Step 1: Filtering traffic...")
            if not run_tshark(self.pcap_path, self.filtered_pcap, filter_rule=self.filter_rule):
                raise Exception("tshark filtering failed")

            # Check if the filtered file is empty
            if not self.filtered_pcap.exists() or self.filtered_pcap.stat().st_size == 0:
                logger.warning(f"{self.pcap_name} is empty after filtering, skipping")
                return False

            # 3. Split into flows
            logger.info(f"Step 2: Splitting flows...")
            if not run_splitcap(self.filtered_pcap, self.flows_dir):
                raise Exception("splitcap splitting failed")

            # 4. Collect flow information
            logger.info(f"Step 3: Collecting flow information...")
            flows = collect_flows(self.flows_dir)

            if not flows:
                logger.warning(f"{self.pcap_name} contains no valid flows, skipping")
                return False

            # 5. Split into train, validation, and test sets based on configuration
            logger.info(f"Step 4: Splitting train/val/test sets (Mode: {SPLIT_MODE})...")
            logging.info(f"Split ratios - Train: {self.train_ratio}, Val: {self.val_ratio}, Test: {1 - self.train_ratio - self.val_ratio}")
            
            if SPLIT_MODE == 'random':
                train_flows, val_flows, test_flows = split_flows_randomly(
                    flows, self.train_ratio, self.val_ratio, seed=RANDOM_SEED
                )
            else:  # Default: Split by timestamp
                train_flows, val_flows, test_flows = split_flows_by_time(
                    flows, self.train_ratio, self.val_ratio
                )

            # 6. Save to corresponding directories (using pcap name as prefix)
            logger.info(f"Step 5: Saving flow files...")
            train_count = save_flows_to_directory_with_prefix(train_flows, self.train_dir, prefix=self.pcap_name)
            val_count = save_flows_to_directory_with_prefix(val_flows, self.val_dir, prefix=self.pcap_name)
            test_count = save_flows_to_directory_with_prefix(test_flows, self.test_dir, prefix=self.pcap_name)

            logger.info(f"Processing complete: {self.category_name}/{self.pcap_name}")
            logger.info(f"  - Total flows: {len(flows)}")
            logger.info(f"  - Train set: {train_count}")
            logger.info(f"  - Val set: {val_count}")
            logger.info(f"  - Test set: {test_count}")

            return True

        except Exception as e:
            logger.error(f"Error processing {self.pcap_name}: {e}")
            return False

        finally:
            # Clean up temporary files
            self.cleanup_temp()


def process_single_pcap(category_name, pcap_path, output_base_dir, 
                        train_ratio=0.7, val_ratio=0.15,
                        filter_rule="tcp or udp"):
    """Helper function to process a single PCAP file"""
    processor = PcapProcessor(category_name, pcap_path, output_base_dir, 
                              train_ratio=train_ratio, val_ratio=val_ratio, 
                              filter_rule=filter_rule)
    return processor.process()