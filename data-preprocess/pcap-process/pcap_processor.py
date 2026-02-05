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
    """处理单个PCAP文件的类"""

    def __init__(self, category_name, pcap_path, output_base_dir,train_ratio=0.7, val_ratio=0.15, filter_rule="tcp or udp"):
        self.category_name = category_name
        self.pcap_path = Path(pcap_path)
        self.pcap_name = self.pcap_path.stem  # 不带扩展名的文件名
        self.output_base_dir = Path(output_base_dir)
        self.train_ratio = train_ratio 
        self.val_ratio = val_ratio
        
        # 修改输出目录结构：train/val/test -> category -> pcap_files
        self.train_dir = self.output_base_dir / "train" / category_name
        self.val_dir = self.output_base_dir / "val" / category_name
        self.test_dir = self.output_base_dir / "test" / category_name

        # 临时目录
        self.temp_base = TEMP_DIR / f"{category_name}_{self.pcap_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.filtered_pcap = self.temp_base / "filtered.pcap"
        self.flows_dir = self.temp_base / "flows"
        self.filter_rule = filter_rule
    def setup_directories(self):
        """创建必要的目录"""
        for dir_path in [self.train_dir, self.val_dir, self.test_dir, self.temp_base, self.flows_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def cleanup_temp(self):
        """清理临时文件"""
        if self.temp_base.exists():
            shutil.rmtree(self.temp_base)

    def process(self):
        """处理PCAP文件的主流程"""
        logger.info(f"开始处理: {self.category_name}/{self.pcap_name}")

        try:
            # 1. 创建目录
            self.setup_directories()

            # 2. 过滤流量
            logger.info(f"步骤1: 过滤流量...")
            if not run_tshark(self.pcap_path, self.filtered_pcap, filter_rule=self.filter_rule):
                raise Exception("tshark过滤失败")

            # 检查过滤后的文件是否为空
            if not self.filtered_pcap.exists() or self.filtered_pcap.stat().st_size == 0:
                logger.warning(f"{self.pcap_name} 过滤后为空，跳过")
                return False

            # 3. 切分成流
            logger.info(f"步骤2: 切分流...")
            if not run_splitcap(self.filtered_pcap, self.flows_dir):
                raise Exception("splitcap切分失败")

            # 4. 收集流信息
            logger.info(f"步骤3: 收集流信息...")
            flows = collect_flows(self.flows_dir)

            if not flows:
                logger.warning(f"{self.pcap_name} 没有有效的流，跳过")
                return False

            # 5. 按配置的方式切分训练集、验证集和测试集
            logger.info(f"步骤4: 切分训练/验证/测试集 (模式: {SPLIT_MODE})...")
            logging.info(f"切分比例 - 训练: {self.train_ratio}, 验证: {self.val_ratio}, 测试: {1 - self.train_ratio - self.val_ratio}")
            
            if SPLIT_MODE == 'random':
                train_flows, val_flows, test_flows = split_flows_randomly(
                    flows, self.train_ratio, self.val_ratio, seed=RANDOM_SEED
                )
            else:  # 默认按时间切分
                train_flows, val_flows, test_flows = split_flows_by_time(
                    flows, self.train_ratio, self.val_ratio
                )

            # 6. 保存到对应目录（使用pcap名称作为前缀）
            logger.info(f"步骤5: 保存流文件...")
            train_count = save_flows_to_directory_with_prefix(train_flows, self.train_dir, prefix=self.pcap_name)
            val_count = save_flows_to_directory_with_prefix(val_flows, self.val_dir, prefix=self.pcap_name)
            test_count = save_flows_to_directory_with_prefix(test_flows, self.test_dir, prefix=self.pcap_name)

            logger.info(f"处理完成: {self.category_name}/{self.pcap_name}")
            logger.info(f"  - 总流数: {len(flows)}")
            logger.info(f"  - 训练集: {train_count}")
            logger.info(f"  - 验证集: {val_count}")
            logger.info(f"  - 测试集: {test_count}")

            return True

        except Exception as e:
            logger.error(f"处理 {self.pcap_name} 时出错: {e}")
            return False

        finally:
            # 清理临时文件
            self.cleanup_temp()



def process_single_pcap(category_name, pcap_path, output_base_dir, 
                        train_ratio=0.7, val_ratio=0.15,
                        filter_rule="tcp or udp"):
    """处理单个PCAP文件的便捷函数"""
    # 传递参数给构造函数
    processor = PcapProcessor(category_name, pcap_path, output_base_dir, 
                              train_ratio=train_ratio, val_ratio=val_ratio, 
                              filter_rule=filter_rule)
    return processor.process()