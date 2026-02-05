import os
from pathlib import Path

# 训练集/验证集/测试集切分比例
TRAIN_RATIO = 0.7   # 70% 训练集
VAL_RATIO = 0.15    # 15% 验证集
TEST_RATIO = 0.15   # 15% 测试集
# 注意: TRAIN_RATIO + VAL_RATIO + TEST_RATIO 应该等于 1.0

# 切分方式: 'time' 或 'random'
SPLIT_MODE = 'random'  # 'time': 按时间顺序切分, 'random': 随机切分

# 随机种子（用于随机切分时保证可重复性）
RANDOM_SEED = 42

# 临时文件目录
TEMP_DIR = Path("./temp")

# 日志配置
LOG_DIR = Path("./logs")
LOG_LEVEL = "INFO"  # 可选: DEBUG, INFO, WARNING, ERROR

# 处理配置
MIN_FLOW_PACKETS = 0  # 最少包数量的流才保留 VPN = 0
MAX_FLOW_SIZE = 100 * 1024 * 1024  # 最大流大小 100MB

# 类别过滤配置
MIN_CATEGORY_FLOWS = 25  # 类别最小流数阈值，流数少于此值的类别将被删除（0表示不过滤）

# 确保必要目录存在
for dir_path in [TEMP_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)