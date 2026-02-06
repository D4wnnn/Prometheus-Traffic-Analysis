import os
from pathlib import Path

# Split ratios for training/validation/test sets
TRAIN_RATIO = 0.7   # 70% Training set
VAL_RATIO = 0.15    # 15% Validation set
TEST_RATIO = 0.15   # 15% Test set
# Note: TRAIN_RATIO + VAL_RATIO + TEST_RATIO should sum to 1.0

# Split mode: 'time' or 'random'
SPLIT_MODE = 'random'  # 'time': sequential split by time, 'random': shuffled split

# Random seed for reproducibility
RANDOM_SEED = 42

# Temporary file directory
TEMP_DIR = Path("./temp")

# Logging configuration
LOG_DIR = Path("./logs")
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR

# Processing configuration
MIN_FLOW_PACKETS = 0  # Minimum packet count to keep a flow (VPN = 0)
MAX_FLOW_SIZE = 100 * 1024 * 1024  # Maximum flow size (100MB)

# Category filtering configuration
MIN_CATEGORY_FLOWS = 25  # Minimum flow count threshold per category; categories below this will be removed (0 means no filtering)

# Ensure necessary directories exist
for dir_path in [TEMP_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)