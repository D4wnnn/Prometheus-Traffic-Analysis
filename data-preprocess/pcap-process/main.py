"""
加密流量数据集预处理主程序
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

# 设置主日志器
log_file = LOG_DIR / f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger = setup_logger("main", log_file)


def clear_directory(dir_path: Path):
    """
    如果目录存在且不为空，则清空其中的所有内容。
    """
    if dir_path.exists() and any(dir_path.iterdir()):
        logger.warning(f"输出目录 '{dir_path}' 非空，即将清空...")
        for item in dir_path.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                    logger.info(f"已删除子目录: {item}")
                else:
                    item.unlink()
                    logger.info(f"已删除文件: {item}")
            except Exception as e:
                logger.error(f"删除 {item} 失败: {e}")
        logger.info(f"目录 '{dir_path}' 已成功清空。")
    elif dir_path.exists():
        logger.info(f"输出目录 '{dir_path}' 为空，无需操作。")
    else:
        logger.info(f"输出目录 '{dir_path}' 不存在，将在后续步骤中创建。")


class DatasetPreprocessor:
    """数据集预处理器主类"""

    def __init__(self, input_dir, output_dir, train_ratio, val_ratio, test_ratio, min_category_flows=0,filter_rule="tcp or udp"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.filter_rule = filter_rule
        self.min_category_flows = min_category_flows  # 类别最小流数阈值
        self.stats = {
            "total_categories": 0,
            "total_pcaps": 0,
            "successful_pcaps": 0,
            "failed_pcaps": 0,
            "category_stats": {},
            "train_total": 0,
            "val_total": 0,
            "test_total": 0,
            "removed_categories": [],  # 记录被删除的类别
        }

    def get_categories(self):
        """获取所有类别（子文件夹）"""
        categories = [d for d in self.input_dir.iterdir() if d.is_dir()]
        return sorted(categories)

    def process_category(self, category_dir):
        """处理一个类别下的所有PCAP文件"""
        category_name = category_dir.name
        logger.info(f"=" * 50)
        logger.info(f"开始处理类别: {category_name}")

        # 获取该类别下的所有pcap文件
        pcap_files = get_pcap_files(category_dir)

        if not pcap_files:
            logger.warning(f"类别 {category_name} 下没有找到PCAP文件")
            return

        logger.info(f"找到 {len(pcap_files)} 个PCAP文件")

        # 初始化统计
        success_count = 0
        fail_count = 0

        # 处理每个pcap文件
        for i, pcap_file in enumerate(pcap_files, 1):
            logger.info(f"-" * 40)
            logger.info(f"处理进度: [{i}/{len(pcap_files)}] {pcap_file.name}")

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
                logger.error(f"处理PCAP文件失败: {e}")
                fail_count += 1
                self.stats["failed_pcaps"] += 1

        # 更新类别统计
        self.stats["category_stats"][category_name] = {
            "total": len(pcap_files),
            "success": success_count,
            "failed": fail_count,
        }

        logger.info(
            f"类别 {category_name} 处理完成: 成功 {success_count}/{len(pcap_files)}"
        )

    def get_category_flow_count(self, category_name):
        """获取某个类别在所有split中的流总数"""
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
        """删除流数目少于阈值的类别"""
        if self.min_category_flows <= 0:
            logger.info("MIN_CATEGORY_FLOWS <= 0，跳过小类别删除")
            return

        logger.info(f"=" * 50)
        logger.info(f"开始检查并删除流数少于 {self.min_category_flows} 的类别...")

        # 获取所有类别
        train_dir = self.output_dir / "train"
        if not train_dir.exists():
            logger.warning("训练集目录不存在，跳过小类别删除")
            return

        categories_to_remove = []

        # 检查每个类别的流数
        for category_dir in train_dir.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                total_flows = self.get_category_flow_count(category_name)

                if total_flows < self.min_category_flows:
                    categories_to_remove.append((category_name, total_flows))
                    logger.warning(
                        f"类别 '{category_name}' 流数 ({total_flows}) 少于阈值 ({self.min_category_flows})，将被删除"
                    )

        # 删除小类别
        for category_name, flow_count in categories_to_remove:
            try:
                # 删除 train/val/test 中的该类别目录
                for split in ["train", "val", "test"]:
                    cat_dir = self.output_dir / split / category_name
                    if cat_dir.exists():
                        shutil.rmtree(cat_dir)
                        logger.info(f"已删除: {cat_dir}")

                # 记录被删除的类别
                self.stats["removed_categories"].append({
                    "name": category_name,
                    "flow_count": flow_count,
                    "reason": f"流数 ({flow_count}) < 阈值 ({self.min_category_flows})"
                })

                # 从类别统计中移除
                if category_name in self.stats["category_stats"]:
                    del self.stats["category_stats"][category_name]

            except Exception as e:
                logger.error(f"删除类别 '{category_name}' 失败: {e}")

        if categories_to_remove:
            logger.info(f"共删除 {len(categories_to_remove)} 个小类别")
            # 更新类别总数
            self.stats["total_categories"] -= len(categories_to_remove)
        else:
            logger.info("没有需要删除的小类别")

    def count_output_flows(self):
        """统计输出目录中的流文件数量"""
        train_dir = self.output_dir / "train"
        val_dir = self.output_dir / "val"
        test_dir = self.output_dir / "test"

        train_count = 0
        val_count = 0
        test_count = 0

        # 统计训练集
        if train_dir.exists():
            for category_dir in train_dir.iterdir():
                if category_dir.is_dir():
                    flows = list(category_dir.glob("*.pcap")) + list(
                        category_dir.glob("*.cap")
                    )
                    train_count += len(flows)

        # 统计验证集
        if val_dir.exists():
            for category_dir in val_dir.iterdir():
                if category_dir.is_dir():
                    flows = list(category_dir.glob("*.pcap")) + list(
                        category_dir.glob("*.cap")
                    )
                    val_count += len(flows)

        # 统计测试集
        if test_dir.exists():
            for category_dir in test_dir.iterdir():
                if category_dir.is_dir():
                    flows = list(category_dir.glob("*.pcap")) + list(
                        category_dir.glob("*.cap")
                    )
                    test_count += len(flows)

        self.stats["train_total"] = train_count
        self.stats["val_total"] = val_count
        self.stats["test_total"] = test_count

        return train_count, val_count, test_count

    def process_all(self):
        """处理所有类别"""
        categories = self.get_categories()

        if not categories:
            logger.error(f"输入目录 {self.input_dir} 下没有找到类别文件夹")
            return

        logger.info(f"找到 {len(categories)} 个类别")
        self.stats["total_categories"] = len(categories)

        # 处理每个类别
        for category_dir in categories:
            self.process_category(category_dir)
            self.stats["total_pcaps"] = (
                self.stats["successful_pcaps"] + self.stats["failed_pcaps"]
            )

        # 删除流数过少的类别
        self.remove_small_categories()

        # 统计输出流文件（在删除小类别之后）
        self.count_output_flows()

        # 输出总体统计
        self.print_stats()
        self.save_stats()

    def print_stats(self):
        """打印统计信息"""
        logger.info("=" * 50)
        logger.info("处理完成 - 总体统计:")
        logger.info(f"  类别总数: {self.stats['total_categories']}")
        logger.info(f"  PCAP总数: {self.stats['total_pcaps']}")
        logger.info(f"  成功处理: {self.stats['successful_pcaps']}")
        logger.info(f"  处理失败: {self.stats['failed_pcaps']}")
        
        # 打印被删除的类别
        if self.stats["removed_categories"]:
            logger.info(f"\n  被删除的小类别 ({len(self.stats['removed_categories'])} 个):")
            for removed in self.stats["removed_categories"]:
                logger.info(f"    - {removed['name']}: {removed['reason']}")
        
        logger.info(f"\n  输出流统计:")
        logger.info(f"    训练集流: {self.stats['train_total']}")
        logger.info(f"    验证集流: {self.stats['val_total']}")
        logger.info(f"    测试集流: {self.stats['test_total']}")
        logger.info(
            f"    总流数: {self.stats['train_total'] + self.stats['val_total'] + self.stats['test_total']}"
        )

        logger.info("\n各类别统计:")
        for category, cat_stats in self.stats["category_stats"].items():
            logger.info(
                f"  {category}: 成功 {cat_stats['success']}/{cat_stats['total']}"
            )

            # 统计该类别的输出流数
            train_cat_dir = self.output_dir / "train" / category
            val_cat_dir = self.output_dir / "val" / category
            test_cat_dir = self.output_dir / "test" / category

            train_flows = 0
            val_flows = 0
            test_flows = 0

            if train_cat_dir.exists():
                train_flows = len(
                    list(train_cat_dir.glob("*.pcap"))
                    + list(train_cat_dir.glob("*.cap"))
                )
            if val_cat_dir.exists():
                val_flows = len(
                    list(val_cat_dir.glob("*.pcap"))
                    + list(val_cat_dir.glob("*.cap"))
                )
            if test_cat_dir.exists():
                test_flows = len(
                    list(test_cat_dir.glob("*.pcap")) + list(test_cat_dir.glob("*.cap"))
                )

            logger.info(f"    - 训练流: {train_flows}, 验证流: {val_flows}, 测试流: {test_flows}")

    def save_stats(self):
        """保存统计信息到文件"""
        stats_file = self.output_dir / "preprocessing_stats.json"

        # 添加输出目录结构信息
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
        logger.info(f"统计信息已保存到: {stats_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="加密流量数据集预处理工具")
    parser.add_argument(
        "--input", type=str, help="输入数据集根目录"
    )
    parser.add_argument(
        "--output", type=str, help="输出数据集根目录"
    )
    parser.add_argument("--category", type=str, help="只处理指定类别")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="如果指定，则在开始前清空输出目录",
    )
    parser.add_argument(
        "--min-category-flows",
        type=int,
        default=MIN_CATEGORY_FLOWS,
        help=f"类别最小流数阈值，流数少于此值的类别将被删除（默认: {MIN_CATEGORY_FLOWS}，0表示不过滤）",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO,
        help=f"训练集比例（默认: {TRAIN_RATIO}）",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VAL_RATIO,
        help=f"验证集比例（默认: {VAL_RATIO}）",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=TEST_RATIO,
        help=f"测试集比例（默认: {TEST_RATIO}）",
    )
    parser.add_argument(
        "--filter-rule",
        type=str,
        default="tcp or udp",
        help=f"tshark 过滤规则（默认: 'tcp or udp'）",
    )

    args = parser.parse_args()

    # 检查输入目录
    if not Path(args.input).exists():
        logger.error(f"输入目录不存在: {args.input}")
        sys.exit(1)
    output_dir = Path(args.output)
    if args.clear:
        clear_directory(output_dir)
    # 创建输出目录
    Path(args.output).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info("加密流量数据集预处理工具")
    logger.info(f"输入目录: {args.input}")
    logger.info(f"输出目录: {args.output}")
    logger.info(f"输出结构: train/val/test -> category -> pcap_prefix_flows")
    logger.info(f"切分模式: {SPLIT_MODE}")
    logger.info(f"切分比例: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    logger.info(f"类别最小流数阈值: {args.min_category_flows}")
    logger.info("=" * 50)

    # 创建预处理器并执行
    preprocessor = DatasetPreprocessor(
        args.input, args.output, args.train_ratio, args.val_ratio, args.test_ratio, min_category_flows=args.min_category_flows,filter_rule=args.filter_rule
    )

    if args.category:
        # 只处理指定类别
        category_dir = Path(args.input) / args.category
        if category_dir.exists():
            preprocessor.process_category(category_dir)
            preprocessor.remove_small_categories()  # 检查并删除小类别
            preprocessor.count_output_flows()
            preprocessor.print_stats()
            preprocessor.save_stats()
        else:
            logger.error(f"指定的类别不存在: {args.category}")
            sys.exit(1)
    else:
        # 处理所有类别
        preprocessor.process_all()

    logger.info("=" * 50)
    logger.info("所有处理已完成!")


if __name__ == "__main__":
    main()