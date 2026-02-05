"""
CIC-IDS2017 数据集专用预处理脚本
逻辑：PCAP切分 -> 读取CSV匹配标签 -> 按比例划分训练/验证/测试集
"""

import os
import shutil
import argparse
import pandas as pd
import subprocess
from pathlib import Path
from tqdm import tqdm
import random

# 复用你现有的配置和工具
from config import (
    LOG_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, 
    TEMP_DIR, RANDOM_SEED
)
from utils import setup_logger, clean_directory

# 设置日志
logger = setup_logger("cicids_processor", LOG_DIR / "cicids_process.log")

class CicIdsPreprocessor:
    def __init__(self, pcap_path, csv_path, output_dir):
        self.pcap_path = Path(pcap_path).resolve()
        self.csv_path = Path(csv_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.pool_dir = TEMP_DIR / "cicids_pool"  # 临时存放切分后的所有流

        # 确保输出目录存在
        self.setup_dirs()

    def setup_dirs(self):
        """创建基础目录结构"""
        for split in ['train', 'val', 'test']:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)
        
        # 清理并重建临时池
        clean_directory(self.pool_dir)

    def run_pkt2flow(self):
        """
        使用 pkt2flow 切分大文件 (比 SplitCap 处理大文件更快)
        如果没有 pkt2flow，请先安装: sudo apt install pkt2flow
        """
        logger.info(f"正在切分原始 PCAP: {self.pcap_path}")
        logger.info(f"这可能需要几分钟，输出目录: {self.pool_dir}")
        
        cmd = [
            "pkt2flow",
            "-u",           # 处理 UDP
            "-o", str(self.pool_dir),
            str(self.pcap_path)
        ]
        
        try:
            # 使用 subprocess 调用
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info("PCAP 切分完成")
        except FileNotFoundError:
            logger.error("未找到 pkt2flow 命令，请先安装: sudo apt install pkt2flow")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"pkt2flow 运行失败: {e}")
            raise

    # def get_pcap_path_from_pool(self, src_ip, dst_ip, src_port, dst_port, proto):
    #     """在池中查找对应的 PCAP 文件"""
    #     # 协议转换 CSV(Int) -> pkt2flow(Str)
    #     if proto == 6:
    #         proto_name = "tcp"
    #     elif proto == 17:
    #         proto_name = "udp"
    #     else:
    #         return None # 忽略其他协议

    #     # 构造文件名模式 (pkt2flow 通常是 srcIP_srcPort_dstIP_dstPort.pcap)
    #     # 注意：CSV 里的方向可能和文件名里的方向是反的，所以要查两次
        
    #     # 正向
    #     name_fwd = f"{src_ip}_{src_port}_{dst_ip}_{dst_port}.pcap"
    #     path_fwd = self.pool_dir / proto_name / name_fwd
        
    #     if path_fwd.exists():
    #         return path_fwd
        
    #     # 反向
    #     name_bwd = f"{dst_ip}_{dst_port}_{src_ip}_{src_port}.pcap"
    #     path_bwd = self.pool_dir / proto_name / name_bwd
        
    #     if path_bwd.exists():
    #         return path_bwd
            
    #     return None
    # def get_pcap_path_from_pool(self, src_ip, dst_ip, src_port, dst_port, proto):
    #     """
    #     在池中查找对应的 PCAP 文件
    #     修复：
    #     1. 增加对 tcp_nosyn 文件夹的支持（处理无握手的截断流）
    #     2. 增加对文件名末尾时间戳的模糊匹配 (*.pcap)
    #     """
    #     # 1. 确定搜索目录列表
    #     search_subdirs = []
    #     if proto == 6:
    #         # TCP 流可能在 tcp 或 tcp_nosyn 目录下
    #         search_subdirs = ["tcp", "tcp_nosyn"]
    #     elif proto == 17:
    #         search_subdirs = ["udp"]
    #     else:
    #         # 其他协议尝试找同名文件夹，或者是 other
    #         search_subdirs = ["other", str(proto)]

    #     # 2. 构造模糊匹配模式 (pkt2flow 格式: src_sport_dst_dport_timestamp.pcap)
    #     # 注意：使用下划线结尾 _*.pcap 以匹配时间戳
    #     pattern_fwd = f"{src_ip}_{src_port}_{dst_ip}_{dst_port}_*.pcap"
    #     pattern_bwd = f"{dst_ip}_{dst_port}_{src_ip}_{src_port}_*.pcap"

    #     # 3. 遍历目录查找
    #     for subdir in search_subdirs:
    #         target_dir = self.pool_dir / subdir
            
    #         if not target_dir.exists():
    #             continue

    #         # 尝试正向匹配
    #         # glob 返回的是 generator，转 list 取第一个
    #         matches = list(target_dir.glob(pattern_fwd))
    #         if matches:
    #             return matches[0]
            
    #         # 尝试反向匹配
    #         matches = list(target_dir.glob(pattern_bwd))
    #         if matches:
    #             return matches[0]

    #     return None
    def get_pcap_path_from_pool(self, src_ip, dst_ip, src_port, dst_port, proto):
        """
        全目录递归搜索 (Nuclear Option)
        不再依赖协议文件夹名称，直接在 temp/cicids_pool 下递归寻找五元组匹配
        """
        # 1. 构造匹配模式 (pkt2flow 格式: IP_Port_IP_Port_timestamp.pcap)
        # 强制转为字符串并去除空格
        src_ip = str(src_ip).strip()
        dst_ip = str(dst_ip).strip()
        src_port = str(int(src_port)) # 关键：防止 80.0
        dst_port = str(int(dst_port)) # 关键：防止 443.0
        
        # 模式 A: A -> B
        pattern_fwd = f"{src_ip}_{src_port}_{dst_ip}_{dst_port}_*.pcap"
        # 模式 B: B -> A
        pattern_bwd = f"{dst_ip}_{dst_port}_{src_ip}_{src_port}_*.pcap"

        # 2. 使用 rglob (Recursive Glob) 在 pool_dir 及其所有子目录中查找
        # 这比指定目录慢一点，但绝对不会漏掉文件
        try:
            # 搜索正向
            matches = list(self.pool_dir.rglob(pattern_fwd))
            if matches:
                return matches[0]
            
            # 搜索反向
            matches = list(self.pool_dir.rglob(pattern_bwd))
            if matches:
                return matches[0]
                
        except Exception as e:
            # 某些极其特殊的文件名可能会导致 glob 报错
            return None

        return None

    def process_and_distribute(self):
        """核心逻辑：读取 CSV -> 匹配文件 -> 分发到 train/val/test"""
        
        # 1. 切分流量 (检测是否已有文件)
        # 简单的非空检查，防止重复切分
        has_files = False
        for _ in self.pool_dir.rglob("*.pcap"):
            has_files = True
            break
            
        if not has_files:
            self.run_pkt2flow()
        else:
            logger.info("检测到临时目录已有文件，跳过切分步骤...")

        # 2. 读取 CSV
        logger.info(f"正在读取 CSV 标签: {self.csv_path}")
        # explicit encoding to avoid issues
        try:
            df = pd.read_csv(self.csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(self.csv_path, encoding='latin1')
            
        # 清洗列名
        df.columns = [c.strip() for c in df.columns]
        
        # === 新增：打印 CSV 样例数据用于调试 ===
        logger.info(f"CSV 样例数据 (前1行):\n{df.iloc[0]}")
        # ====================================
        
        grouped = df.groupby('Label')
        
        total_matched = 0
        total_missing = 0

        for label, group_df in grouped:
            clean_label = label.replace(' ', '_').replace('–', '-').replace('/', '-')
            logger.info(f"正在处理类别: {clean_label} (共 {len(group_df)} 条记录)")

            valid_files = []
            
            # 使用 tqdm 显示进度
            for _, row in tqdm(group_df.iterrows(), total=len(group_df), desc=f"Matching {clean_label}"):
                try:
                    # 强转类型，处理可能的 NaN
                    if pd.isna(row['Source Port']) or pd.isna(row['Destination Port']):
                        continue
                        
                    f_path = self.get_pcap_path_from_pool(
                        row['Source IP'], row['Destination IP'], 
                        row['Source Port'], row['Destination Port'], 
                        row['Protocol']
                    )
                    
                    if f_path:
                        flow_id = row['Flow ID'].replace(' ', '') # 清洗 Flow ID
                        valid_files.append((f_path, flow_id))
                    else:
                        total_missing += 1
                        
                        # === 调试：只打印前 3 个失败的案例 ===
                        if total_missing <= 3:
                            logger.warning(f"缺失文件样本: Src={row['Source IP']}:{row['Source Port']} -> Dst={row['Destination IP']}:{row['Destination Port']}")
                        # ==================================
                        
                except Exception as e:
                    continue

            total_matched += len(valid_files)
            
            if not valid_files:
                logger.warning(f"类别 {clean_label} 没有匹配到任何文件！")
                continue
            
            # --- 以下部分保持不变 (Shuffle & Split) ---
            random.seed(RANDOM_SEED)
            random.shuffle(valid_files)

            n = len(valid_files)
            train_end = int(n * TRAIN_RATIO)
            val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
            
            splits = {
                'train': valid_files[:train_end],
                'val':   valid_files[train_end:val_end],
                'test':  valid_files[val_end:]
            }

            for split_name, files in splits.items():
                dest_dir = self.output_dir / split_name / clean_label
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                for src_pcap, flow_id in files:
                    new_name = f"{src_pcap.stem}_{flow_id}.pcap"
                    shutil.copy2(src_pcap, dest_dir / new_name)
            
            logger.info(f"类别 {clean_label} 完成: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

        logger.info("="*50)
        logger.info(f"处理结束. 匹配成功: {total_matched}, 匹配失败: {total_missing}")
    def process_and_distribute(self):
        """核心逻辑：读取 CSV -> 匹配文件 -> 分发到 train/val/test"""
        
        # 1. 切分流量
        if not any(self.pool_dir.iterdir()): # 如果池子是空的才切分，避免重复跑
            self.run_pkt2flow()
        else:
            logger.info("检测到临时目录已有文件，跳过切分步骤...")

        # 2. 读取 CSV
        logger.info(f"正在读取 CSV 标签: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        # 清洗列名（去除空格）
        df.columns = [c.strip() for c in df.columns]
        
        # 3. 按标签分组处理
        # 比如 groups 包含 ('BENIGN', dataframe), ('DDoS', dataframe)
        grouped = df.groupby('Label')
        
        total_matched = 0
        total_missing = 0

        for label, group_df in grouped:
            # 某些标签可能包含特殊字符，建议清洗，例如 "Web Attack - XSS" -> "WebAttack_XSS"
            clean_label = label.replace(' ', '_').replace('–', '-').replace('/', '-')
            logger.info(f"正在处理类别: {clean_label} (共 {len(group_df)} 条记录)")

            # 获取该类别下所有有效的 pcap 文件路径
            valid_files = []
            
            for _, row in tqdm(group_df.iterrows(), total=len(group_df), desc=f"Matching {clean_label}"):
                try:
                    f_path = self.get_pcap_path_from_pool(
                        row['Source IP'], row['Destination IP'], 
                        row['Source Port'], row['Destination Port'], 
                        row['Protocol']
                    )
                    if f_path:
                        # 记录 Flow ID 以防重名 (IP-IP-Port-Port-Proto)
                        flow_id = row['Flow ID']
                        valid_files.append((f_path, flow_id))
                    else:
                        total_missing += 1
                except Exception:
                    continue

            total_matched += len(valid_files)
            
            if not valid_files:
                logger.warning(f"类别 {clean_label} 没有匹配到任何文件！")
                continue

            # 4. 随机打乱
            random.seed(RANDOM_SEED)
            random.shuffle(valid_files)

            # 5. 计算切分点
            n = len(valid_files)
            train_end = int(n * TRAIN_RATIO)
            val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
            
            splits = {
                'train': valid_files[:train_end],
                'val':   valid_files[train_end:val_end],
                'test':  valid_files[val_end:]
            }

            # 6. 移动文件
            for split_name, files in splits.items():
                dest_dir = self.output_dir / split_name / clean_label
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                for src_pcap, flow_id in files:
                    # 重命名以包含 FlowID，避免冲突并保留信息
                    # 格式: OriginalName_FlowID.pcap
                    new_name = f"{src_pcap.stem}_{flow_id}.pcap"
                    shutil.copy2(src_pcap, dest_dir / new_name)
            
            logger.info(f"类别 {clean_label} 完成: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

        logger.info("="*50)
        logger.info(f"处理结束. 匹配成功: {total_matched}, 匹配失败(文件太小被丢弃): {total_missing}")
        
        # 7. 清理临时文件 (可选)
        # shutil.rmtree(self.pool_dir) 

def main():
    parser = argparse.ArgumentParser(description="CIC-IDS2017 专用预处理工具")
    parser.add_argument("--pcap", type=str, required=True, help="原始大 PCAP 文件路径 (例如 Friday-Fixed.pcap)")
    parser.add_argument("--csv", type=str, required=True, help="对应的 CSV 标签文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出数据集根目录")
    parser.add_argument("--clear", action="store_true", help="清空输出目录")
    
    args = parser.parse_args()

    if args.clear:
        clean_directory(Path(args.output))

    processor = CicIdsPreprocessor(args.pcap, args.csv, args.output)
    processor.process_and_distribute()

if __name__ == "__main__":
    main()